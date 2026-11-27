"""Small numerical and gradient tests; run python -m unittest -v."""

import unittest
import torch
from grla import GRLA, ridge_attention


class GRLATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_weighted_ridge_output_and_gradients(self):
        # Independent value-side normal-equation reference, including N < d.
        for tokens in (3, 11):
            inputs = [torch.randn(2, 2, tokens, 5, requires_grad=True) for _ in range(3)]
            inputs += [torch.rand(2, 2, tokens, requires_grad=True),
                       torch.full((2, 1, 1), 0.7, requires_grad=True)]
            q, k, v, beta, lam = inputs
            output = ridge_attention(*inputs)
            a = k.transpose(-1, -2) @ (beta.unsqueeze(-1) * k) + lam * torch.eye(5)
            fitted = torch.linalg.solve(a, k.transpose(-1, -2) @ (beta.unsqueeze(-1) * v))
            reference = q @ fitted
            torch.testing.assert_close(output, reference, atol=3e-5, rtol=3e-5)
            weight = torch.randn_like(output)
            actual_grad = torch.autograd.grad((output * weight).sum(), inputs, retain_graph=True)
            expected_grad = torch.autograd.grad((reference * weight).sum(), inputs)
            for actual, expected in zip(actual_grad, expected_grad):
                torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_zero_gates_and_rank_deficient_keys(self):
        q = torch.randn(1, 2, 3, 8)
        k = torch.ones_like(q)
        v = torch.randn_like(q)
        lam = torch.ones(2, 1, 1)
        out = ridge_attention(q, k, v, torch.zeros(1, 2, 3), lam)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        out = ridge_attention(q, k, v, torch.ones(1, 2, 3), lam)
        self.assertTrue(torch.isfinite(out).all())

    def test_module_and_optional_output_layers(self):
        for norm, gate in ((True, True), (False, False), (True, False), (False, True)):
            model = GRLA(24, 3, use_norm=norm, use_output_gate=gate)
            x = torch.randn(2, 7, 24, requires_grad=True)
            y = model(x)
            self.assertEqual(y.shape, x.shape)
            y.square().mean().backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            torch.testing.assert_close(model.get_lambda_val(), torch.ones(3, 1, 1))

    def test_rope_callback_layout(self):
        model = GRLA(24, 3)
        x = torch.randn(2, 6, 24)
        calls = []
        def identity_rope(t):
            calls.append(tuple(t.shape))
            return t
        torch.testing.assert_close(model(x), model(x, 2, 3, identity_rope))
        self.assertEqual(calls, [(2, 2, 3, 24)] * 2)

    def test_validation(self):
        for args in ((25, 3), (24, 0)):
            with self.assertRaises(ValueError):
                GRLA(*args)
        with self.assertRaises(ValueError):
            GRLA(24, 3, lambda_lower_bound=0)
        with self.assertRaises(ValueError):
            GRLA(24, 3)(torch.randn(1, 6, 24), 2, 2, lambda t: t)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_autocast_backward(self):
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            model = GRLA(32, 4).cuda()
            x = torch.randn(2, 9, 32, device="cuda", requires_grad=True)
            with torch.autocast("cuda", dtype=dtype):
                y = model(x)
            self.assertEqual(y.dtype, dtype)
            y.float().square().mean().backward()
            self.assertTrue(torch.isfinite(y).all())
            self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
