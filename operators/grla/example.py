"""Run a small forward/backward example: python example.py."""

import torch
from grla import GRLA


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attention = GRLA(dim=192, num_heads=6).to(device)
    tokens = torch.randn(2, 196, 192, device=device, requires_grad=True)
    output = attention(tokens)
    output.square().mean().backward()
    print(f"device={device}, input={tuple(tokens.shape)}, output={tuple(output.shape)}")
    print(f"input gradient finite: {torch.isfinite(tokens.grad).all().item()}")
