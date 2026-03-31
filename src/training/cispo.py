"""CISPO — Contrastive loss between aligned and misaligned trajectories."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContrastivePair:
    """A positive-negative trajectory pair for contrastive training."""

    pos_prompt: str
    pos_response: str
    pos_score: float
    neg_prompt: str
    neg_response: str
    neg_score: float


def contrastive_trajectory_loss(
    pos_rewards: "torch.Tensor",
    neg_rewards: "torch.Tensor",
    margin: float = 0.5,
) -> "torch.Tensor":
    """Margin-based contrastive loss.

    For each (pos, neg) pair: loss = max(0, margin - (pos - neg)).
    Returns mean over all pairs.
    """
    import torch

    if pos_rewards.numel() == 0 or neg_rewards.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)

    # All pairs: (n_pos, 1) - (1, n_neg) → (n_pos, n_neg)
    diff = pos_rewards.unsqueeze(1) - neg_rewards.unsqueeze(0)
    losses = torch.clamp(margin - diff, min=0.0)
    return losses.mean()


def infonce_trajectory_loss(
    pos_embeddings: "torch.Tensor",
    neg_embeddings: "torch.Tensor",
    temperature: float = 0.07,
) -> "torch.Tensor":
    """InfoNCE contrastive loss on trajectory embeddings.

    Each positive is an anchor; all negatives are distractors.
    """
    import torch
    import torch.nn.functional as F

    if pos_embeddings.numel() == 0 or neg_embeddings.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Normalize embeddings
    pos_norm = F.normalize(pos_embeddings, dim=-1)
    neg_norm = F.normalize(neg_embeddings, dim=-1)

    # Similarity matrix: (n_pos, n_neg)
    logits = torch.matmul(pos_norm, neg_norm.T) / temperature

    # For each positive, the "correct" class is itself (diagonal)
    # We treat this as a classification problem where positive should be most similar to itself
    # Using self-similarity as positive and cross-similarity as negatives
    n_pos = pos_embeddings.shape[0]

    # Self-similarity (positive pairs)
    pos_self = (pos_norm * pos_norm).sum(dim=-1) / temperature  # (n_pos,)

    # Concatenate: [self_sim, neg_sims] → log_softmax → take index 0
    all_logits = torch.cat([pos_self.unsqueeze(1), logits], dim=1)  # (n_pos, 1+n_neg)
    labels = torch.zeros(n_pos, dtype=torch.long, device=pos_embeddings.device)
    return F.cross_entropy(all_logits, labels)


def build_contrastive_pairs(
    aligned: list,
    misaligned: list,
    *,
    score_threshold: float = 0.6,
) -> list[ContrastivePair]:
    """Build contrastive pairs from aligned vs misaligned trajectories.

    Aligned trajectories with score >= threshold are positives.
    Misaligned trajectories with score < threshold are negatives.
    Pairs all positives with all negatives.
    """
    positives = [t for t in aligned if t.outcome_score >= score_threshold]
    negatives = [t for t in misaligned if t.outcome_score < score_threshold]

    pairs: list[ContrastivePair] = []
    for pos in positives:
        for neg in negatives:
            pairs.append(ContrastivePair(
                pos_prompt=pos.prompt,
                pos_response=pos.response,
                pos_score=pos.outcome_score,
                neg_prompt=neg.prompt,
                neg_response=neg.response,
                neg_score=neg.outcome_score,
            ))
    return pairs
