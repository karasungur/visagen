
import pytest
import torch

from visagen.training.optimizers.adabelief import AdaBelief


class TestAdaBelief:
    def test_init_defaults(self):
        params = [torch.tensor([1.0], requires_grad=True)]
        opt = AdaBelief(params)
        assert opt.defaults['lr'] == 1e-3
        assert opt.defaults['betas'] == (0.9, 0.999)
        assert opt.defaults['eps'] == 1e-16
        assert opt.defaults['lr_dropout'] == 1.0
        assert opt.defaults['lr_cos_period'] == 0
        assert opt.defaults['clipnorm'] == 0.0

    def test_step_updates_params(self):
        # Simple quadratic function: y = x^2, dy/dx = 2x
        # x_0 = 1.0, grad = 2.0
        x = torch.tensor([1.0], requires_grad=True)
        opt = AdaBelief([x], lr=0.1)

        # Calculate gradients
        loss = x ** 2
        loss.backward()

        # Initial state
        assert x.item() == 1.0
        assert x.grad.item() == 2.0

        # Step
        opt.step()

        # Param should decrease
        assert x.item() < 1.0

    def test_lr_dropout(self):
        # With lr_dropout=0.0, no updates should happen (all dropped)
        # Note: Implementation uses bernoulli(p), so p=0 means 0 probability of 1 (keep), so always 0.
        x = torch.tensor([1.0], requires_grad=True)
        opt = AdaBelief([x], lr=0.1, lr_dropout=0.0)

        loss = x ** 2
        loss.backward()

        opt.step()

        # Should be exactly the same
        assert x.item() == 1.0

    def test_lr_dropout_partial(self):
        # With p=0.5, run many times, should see some updates and some non-updates
        # But for a single tensor, it's all or nothing per element.
        # Let's use a large tensor
        shape = (1000,)
        x = torch.ones(shape, requires_grad=True)
        opt = AdaBelief([x], lr=0.1, lr_dropout=0.5)

        loss = (x ** 2).sum()
        loss.backward()

        opt.step()

        # Check how many elements changed
        # Initial was 1.0. If updated, it should be < 1.0
        changed = (x < 1.0).sum().item()

        # Should be roughly 50%
        assert 400 < changed < 600

    def test_lr_cos_schedule(self):
        # Period = 4.
        # t=0 (step 1): cos(0) = 1 -> lr * 1.0
        # t=1 (step 2): cos(pi/2) = 0 -> lr * 0.5
        # t=2 (step 3): cos(pi) = -1 -> lr * 0.0
        x = torch.tensor([1.0], requires_grad=True)
        lr = 0.1
        period = 4
        opt = AdaBelief([x], lr=lr, lr_cos_period=period)

        # We need to inspect the update magnitude or the effective LR.
        # It's hard to inspect internal vars directly without interfering.
        # Let's inspect the parameter change.

        # Step 1: Max LR
        x.grad = torch.ones_like(x)  # constant grad
        opt.step()
        # change1 = 1.0 - x.item()

        # Reset x but keep optimizer state (steps increment)
        with torch.no_grad():
            x.copy_(torch.tensor([1.0]))

        # Step 2: Half LR (approx, depending on beta/adam behavior)
        # But AdaBelief adapts. If we keep grad constant = 1.
        # exp_avg will accumulate.
        # Let's check step 3 (t=2) where LR multiplier should be 0.

        x.grad = torch.ones_like(x)
        opt.step() # Step 2

        # Step 3: Should be 0 update if cos factor works perfectly
        # cos(2 * 2pi/4) = cos(pi) = -1. (1 + -1)/2 = 0.
        current_val = x.item()
        x.grad = torch.ones_like(x)
        opt.step() # Step 3

        # Check no change
        assert x.item() == pytest.approx(current_val)

    def test_clipnorm(self):
        # Create a gradient with large norm
        x = torch.tensor([10.0, 10.0], requires_grad=True)
        opt = AdaBelief([x], lr=0.1, clipnorm=1.0)

        loss = x.sum()
        loss.backward()

        # Grads are [1.0, 1.0]. Norm is sqrt(2) ≈ 1.414
        # Should be clipped to 1.0.
        # So grads become [1/1.414, 1/1.414] ≈ [0.707, 0.707]

        # We can check the grads *after* step if we inspected them,
        # but step() modifies params.
        # The optimizer modifies grads in-place during `clip_grad_norm_`?
        # Yes, `torch.nn.utils.clip_grad_norm_` modifies grads in-place.

        opt.step()

        # Check if gradients were clipped
        # Note: PyTorch's clip_grad_norm_ modifies .grad attributes in-place
        grad_norm = x.grad.norm().item()
        assert grad_norm == pytest.approx(1.0, rel=1e-3)

    def test_convergence(self):
        # Simple convergence test
        x = torch.tensor([2.0], requires_grad=True)
        opt = AdaBelief([x], lr=0.1)

        for _ in range(100):
            opt.zero_grad()
            loss = x ** 2
            loss.backward()
            opt.step()

        assert x.item() == pytest.approx(0.0, abs=0.1)
