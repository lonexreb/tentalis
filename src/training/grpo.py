"""GRPO math utilities: advantage calculation, clipped loss, KL penalty."""

from __future__ import annotations


def compute_group_advantages(rewards: list[float]) -> list[float]:
    """Normalize rewards within a group: (r_i - mean) / std.

    If std is zero (all rewards identical), returns all zeros.
    """
    n = len(rewards)
    if n < 2:
        return [0.0] * n
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    if variance < 1e-12:
        return [0.0] * n
    std = variance**0.5
    return [(r - mean) / std for r in rewards]


def clipped_surrogate_loss(
    ratios: "torch.Tensor",
    advantages: "torch.Tensor",
    clip_eps: float = 0.2,
) -> "torch.Tensor":
    """PPO/GRPO clipped surrogate objective (negated for gradient descent)."""
    import torch  # noqa: F811

    clipped = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
    surr1 = ratios * advantages
    surr2 = clipped * advantages
    return -torch.min(surr1, surr2).mean()


def kl_penalty(
    log_probs: "torch.Tensor",
    ref_log_probs: "torch.Tensor",
    beta: float = 0.1,
) -> "torch.Tensor":
    """KL divergence penalty: beta * mean(log_pi - log_pi_ref)."""
    return beta * (log_probs - ref_log_probs).mean()


def asymmetric_clipped_surrogate_loss(
    ratios: "torch.Tensor",
    advantages: "torch.Tensor",
    clip_eps: float = 0.2,
    clip_eps_high: float = 0.28,
) -> "torch.Tensor":
    """PPO/GRPO clipped surrogate with asymmetric clipping.

    Uses a higher clip bound for positive advantages to encourage exploration
    of improved responses (OpenClaw-RL style).
    """
    import torch

    clip_low = 1.0 - clip_eps
    clip_high = 1.0 + clip_eps_high
    clipped = torch.clamp(ratios, clip_low, clip_high)
    surr1 = ratios * advantages
    surr2 = clipped * advantages
    return -torch.min(surr1, surr2).mean()


def combined_loss(
    rl_loss: "torch.Tensor",
    opd_loss: "torch.Tensor",
    w_rl: float = 0.7,
    w_opd: float = 0.3,
) -> "torch.Tensor":
    """Combine RL clipped surrogate loss with OPD distillation loss."""
    return w_rl * rl_loss + w_opd * opd_loss


def multi_loss(
    losses: dict[str, "torch.Tensor"],
    weights: dict[str, float],
) -> "torch.Tensor":
    """Weighted combination of multiple named loss terms."""
    import torch

    total = torch.tensor(0.0)
    for name, loss in losses.items():
        w = weights.get(name, 0.0)
        if w > 0:
            total = total + w * loss
    return total
