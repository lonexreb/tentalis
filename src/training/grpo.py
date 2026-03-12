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
