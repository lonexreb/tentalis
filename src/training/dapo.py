"""DAPO — Dynamic Advantage Policy Optimization utilities.

DAPO extends GRPO with:
1. Dynamic sampling: filter low-reward rollouts, promote diversity
2. Entropy bonus: prevent premature convergence
3. Asymmetric clipping: already in grpo.py (asymmetric_clipped_surrogate_loss)
"""

from __future__ import annotations

from collections import defaultdict

from src.events.types import TrainingRolloutEvent


def dynamic_sample_filter(
    rollouts: list[TrainingRolloutEvent],
    *,
    min_reward_threshold: float = 0.1,
) -> list[TrainingRolloutEvent]:
    """DAPO dynamic sampling: filter low-reward rollouts and deduplicate.

    1. Group by prompt.
    2. Within each group, remove rollouts with outcome_score below threshold.
    3. Deduplicate identical responses within each group.
    4. Return filtered set (skip groups left empty after filtering).
    """
    groups: dict[str, list[TrainingRolloutEvent]] = defaultdict(list)
    for r in rollouts:
        groups[r.prompt].append(r)

    filtered: list[TrainingRolloutEvent] = []
    for prompt, group in groups.items():
        # Remove low-reward
        above = [r for r in group if r.outcome_score >= min_reward_threshold]
        if not above:
            continue

        # Deduplicate identical responses
        seen: set[str] = set()
        unique: list[TrainingRolloutEvent] = []
        for r in above:
            if r.response not in seen:
                seen.add(r.response)
                unique.append(r)

        filtered.extend(unique)

    return filtered


def entropy_bonus(
    logits: "torch.Tensor",
    beta_entropy: float = 0.01,
) -> "torch.Tensor":
    """Entropy bonus to prevent premature convergence.

    H(pi) = -sum(pi * log(pi))
    Returns beta * mean_entropy (to be subtracted from loss).
    """
    import torch

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # per-token entropy
    return beta_entropy * entropy.mean()


def dapo_loss(
    ratios: "torch.Tensor",
    advantages: "torch.Tensor",
    logits: "torch.Tensor",
    clip_eps: float = 0.2,
    clip_eps_high: float = 0.28,
    beta_entropy: float = 0.01,
) -> "torch.Tensor":
    """Full DAPO loss: asymmetric clipped surrogate - entropy bonus.

    loss = asymmetric_clip(ratios, advantages) - entropy_bonus(logits)
    """
    from src.training.grpo import asymmetric_clipped_surrogate_loss

    clip_loss = asymmetric_clipped_surrogate_loss(
        ratios, advantages, clip_eps, clip_eps_high,
    )
    ent_bonus = entropy_bonus(logits, beta_entropy)
    return clip_loss - ent_bonus
