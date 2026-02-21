"""
AdaBelief optimizer with legacy DFL features.

Implements AdaBelief algorithm with specific features used in DeepFaceLab (legacy):
- lr_dropout: Randomly drops updates for parameters (stochastic depth-like behavior)
- lr_cos: Cyclical cosine learning rate scheduling
- clipnorm: Global gradient clipping within the optimizer
"""

import math
from collections.abc import Callable
from typing import overload

import torch
from torch.optim import Optimizer


class AdaBelief(Optimizer):
    """
    AdaBelief Optimizer with legacy DFL features.

    AdaBelief adapts the step size according to the "belief" in the gradient direction.
    It views the exponential moving average (EMA) of the noisy gradient as the prediction
    of the gradient at the next time step.

    Legacy Features:
    - lr_dropout: Probability of keeping the update (0.0 to 1.0). If < 1.0, updates
      are randomly skipped for parameters.
    - lr_cos_period: Period for cosine annealing of learning rate in iterations.
      If > 0, LR cycles from lr to 0 and back.
    - clipnorm: Max norm for global gradient clipping. If > 0, gradients are clipped.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate. Default: 1e-3.
        betas: Coefficients for computing running averages of gradient and
            its square. Default: (0.9, 0.999).
        eps: Term added to the denominator to improve numerical stability.
            Default: 1e-16.
        weight_decay: Weight decay (L2 penalty). Default: 0.0.
        lr_dropout: Learning rate dropout probability (keep rate). Default: 1.0 (no dropout).
        lr_cos_period: Period for cosine LR scheduling. Default: 0 (disabled).
        clipnorm: Maximum global norm for gradient clipping. Default: 0.0 (disabled).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
        lr_dropout: float = 1.0,
        lr_cos_period: int = 0,
        clipnorm: float = 0.0,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= lr_dropout <= 1.0:
            raise ValueError(f"Invalid lr_dropout value: {lr_dropout}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "lr_dropout": lr_dropout,
            "lr_cos_period": lr_cos_period,
            "clipnorm": clipnorm,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("lr_dropout", 1.0)
            group.setdefault("lr_cos_period", 0)
            group.setdefault("clipnorm", 0.0)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 1. Global Gradient Clipping (clipnorm)
        # In DFL legacy, this is done before updates.
        for group in self.param_groups:
            clipnorm = group["clipnorm"]
            if clipnorm > 0.0:
                # Collect params for this group
                params_with_grad = [p for p in group["params"] if p.grad is not None]
                if params_with_grad:
                    torch.nn.utils.clip_grad_norm_(params_with_grad, clipnorm)

        for group in self.param_groups:
            lr_dropout = group["lr_dropout"]
            lr_cos_period = group["lr_cos_period"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            # Calculate current LR with Cosine Annealing if enabled
            current_lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update biases
                # bias1 = 1 - beta1 ** step
                # bias2 = 1 - beta2 ** step
                # AdaBelief usually doesn't use bias correction in the paper implementation heavily like Adam?
                # Wait, the official implementation does use bias correction.
                # Let's check DFL legacy implementation.
                # m_t = beta1 * ms + (1-beta1) * g
                # v_t = beta2 * vs + (1-beta2) * (g - m_t)^2
                # v_diff = - lr * m_t / (sqrt(v_t) + eps)
                # new_v = v + v_diff
                #
                # DFL Legacy DOES NOT use bias correction (1-beta^t).
                # It just does: m_t = beta1*m + (1-beta1)*g
                # So we follow DFL Legacy, not PyTorch Adam.

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # AdaBelief: variance of the gradient
                # v_t = beta2 * v_{t-1} + (1-beta2) * (g_t - m_t)^2
                grad_residual = grad - exp_avg
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad_residual, grad_residual, value=1 - beta2
                )

                denom = exp_avg_sq.sqrt().add_(eps)

                # Calculate update step
                # step_size = lr

                # Cosine LR Schedule (Legacy DFL style)
                step_lr = current_lr
                if lr_cos_period > 0:
                    # lr *= (cos( iters * (2pi / period) ) + 1) / 2
                    # Note: Legacy uses `self.iterations` which increments per step.
                    # Here `step` is 1-based (incremented above).
                    # Legacy starts at 0. So use step-1.
                    cos_val = math.cos((step - 1) * (2 * math.pi / lr_cos_period))
                    step_lr *= (cos_val + 1.0) / 2.0

                # Compute update
                # update = - step_lr * exp_avg / denom
                # p.add_(update)
                # But we need to support lr_dropout.

                update = exp_avg / denom

                # Apply LR Dropout (Legacy DFL style)
                if lr_dropout < 1.0:
                    # Random binomial mask
                    # legacy: v_diff *= lr_rnd
                    # mask = random_binomial(shape, p=lr_dropout)
                    mask = torch.bernoulli(torch.full_like(p, lr_dropout))
                    # If mask is 0, update is 0. If 1, update is kept.
                    update.mul_(mask)

                # Apply update
                p.add_(update, alpha=-step_lr)

        return loss
