# MIT License
# 
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2026 Yuxin Mao (GRLA extraction and modifications).
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Global Ridge Linear Attention (GRLA), implemented using only PyTorch.

Adapted from the GRLA LinearAttention implementation in the authors' modified
vision-ttt codebase. The original file credits Dongchen Han. The MIT notice is retained below.
"""

import math
from typing import Callable, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def ridge_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    beta: Tensor,
    regularization: Tensor,
) -> Tensor:
    """Apply the weighted ridge operator to already transformed Q/K.

    Args:
        query, key: [batch, heads, tokens, head_dim]. Query has already been
            activated and key normalized; this function does not transform them.
        value: [batch, heads, tokens, value_dim].
        beta: [batch, heads, tokens], nonnegative token weights.
        regularization: positive ridge coefficients broadcastable to
            [batch, heads, head_dim, head_dim], e.g. [heads, 1, 1].

    Returns:
        [batch, heads, tokens, value_dim], in query.dtype.

    Both statistics use beta. The implementation solves against Q transpose
    without forming an inverse, an N-by-N coefficient map, or a diagonal gate
    matrix. Statistics, solve, and readout use FP32. The solved query is rounded
    to query.dtype before readout, matching the original mixed-precision path.
    """
    if query.ndim != 4 or query.shape != key.shape:
        raise ValueError("query and key must share shape [batch, heads, tokens, head_dim]")
    if value.ndim != 4 or value.shape[:3] != key.shape[:3]:
        raise ValueError("value must share the batch, head, and token dimensions")
    if beta.shape != key.shape[:3]:
        raise ValueError("beta must have shape [batch, heads, tokens]")
    if any(t.device != query.device for t in (key, value, beta, regularization)):
        raise ValueError("all inputs must be on the same device")

    with torch.autocast(device_type=query.device.type, enabled=False):
        q, k = query.float(), key.float()
        weighted_key = k * beta.float().unsqueeze(-1)
        gram = weighted_key.transpose(-1, -2) @ k
        identity = torch.eye(k.shape[-1], device=k.device, dtype=torch.float32)
        system = gram + regularization.float() * identity
        solved_query = torch.linalg.solve(system, q.transpose(-1, -2))
        solved_query = solved_query.transpose(-1, -2).to(query.dtype)
        # Keep the source's separate casts for the two statistics, including
        # their gradient accumulation behavior under mixed precision.
        readout_key = key.float() * beta.float().unsqueeze(-1)
        context = readout_key.transpose(-1, -2) @ value.float()
        output = solved_query.float() @ context
    return output.to(query.dtype)


class GRLA(nn.Module):
    """Projected multi-head GRLA operator for [batch, tokens, dim] inputs.

    Defaults preserve the extracted GRLA configuration: SiLU queries, L2 keys,
    beta weighting, per-head softplus + 0.1 regularization, and output LayerNorm
    and sigmoid gate. The initial effective ridge coefficient is 1.0.

    Set use_norm=False and use_output_gate=False for the projected ridge
    operator without these two output transformations. Neither setting changes
    the beta gates used inside the ridge objective.

    Optional RoPE is supplied by the caller and applied to projected Q/K before
    SiLU/key normalization, in the original [batch, height, width, dim] layout.
    No positional encoding, local convolution, residual block, or MLP is added.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        lambda_lower_bound: float = 0.1,
        lambda_initial_value: float = 1.0,
        use_norm: bool = True,
        use_output_gate: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("dim must be positive and divisible by positive num_heads")
        if not math.isfinite(lambda_lower_bound) or lambda_lower_bound <= 0:
            raise ValueError("lambda_lower_bound must be finite and positive")
        if not math.isfinite(lambda_initial_value) or lambda_initial_value <= lambda_lower_bound:
            raise ValueError("lambda_initial_value must exceed lambda_lower_bound")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.lambda_lower_bound = lambda_lower_bound
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6) if use_norm else None
        self.gate_proj = (
            nn.Sequential(nn.Linear(dim, self.head_dim), nn.Linear(self.head_dim, dim))
            if use_output_gate else None
        )
        self.beta_proj = nn.Linear(dim, num_heads, bias=True)
        initial = torch.tensor(lambda_initial_value - lambda_lower_bound)
        # Match the original parameterization and initialization for the default.
        raw_initial = initial + torch.log(-torch.expm1(-initial))
        self.lambda_ = nn.Parameter(torch.full((num_heads, 1, 1), raw_initial.item()))

    def get_lambda_val(self) -> Tensor:
        return F.softplus(self.lambda_.float()) + self.lambda_lower_bound

    def forward(
        self,
        x: Tensor,
        h: Optional[int] = None,
        w: Optional[int] = None,
        rope: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError("x must have shape [batch, tokens, dim]")
        batch, tokens, channels = x.shape
        if tokens == 0:
            raise ValueError("x must contain at least one token")
        q, k, v = self.qkv(x).split(channels, dim=-1)
        if rope is not None:
            if h is None or w is None or h <= 0 or w <= 0 or h * w != tokens:
                raise ValueError("RoPE requires positive h and w with h * w == tokens")
            q = rope(q.reshape(batch, h, w, channels)).reshape(batch, tokens, channels)
            k = rope(k.reshape(batch, h, w, channels)).reshape(batch, tokens, channels)

        q = q.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        q = F.silu(q)
        k = F.normalize(k, p=2, dim=-1, eps=1e-12)
        out = ridge_attention(q, k, v, beta, self.get_lambda_val())
        out = out.transpose(1, 2).reshape(batch, tokens, channels)
        if self.norm is not None:
            out = self.norm(out)
        if self.gate_proj is not None:
            out = out * torch.sigmoid(self.gate_proj(out))
        return self.proj(out)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"lambda_lower_bound={self.lambda_lower_bound}, "
            f"use_norm={self.norm is not None}, use_output_gate={self.gate_proj is not None}"
        )
