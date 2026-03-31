"""Tests for CISPO contrastive loss functions and pair building."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.training.cispo import ContrastivePair, build_contrastive_pairs


# Mock trajectory for testing (matches StoredTrajectory interface)
@dataclass
class _MockTrajectory:
    prompt: str = "test prompt"
    response: str = "test response"
    outcome_score: float = 0.8


def test_contrastive_pair_fields():
    pair = ContrastivePair(
        pos_prompt="p1", pos_response="r1", pos_score=0.9,
        neg_prompt="p2", neg_response="r2", neg_score=0.1,
    )
    assert pair.pos_score == 0.9
    assert pair.neg_score == 0.1


def test_build_contrastive_pairs():
    aligned = [_MockTrajectory(outcome_score=0.8), _MockTrajectory(outcome_score=0.9)]
    misaligned = [_MockTrajectory(outcome_score=0.2), _MockTrajectory(outcome_score=0.1)]
    pairs = build_contrastive_pairs(aligned, misaligned, score_threshold=0.6)
    assert len(pairs) == 4  # 2 positives x 2 negatives


def test_build_contrastive_pairs_empty_misaligned():
    aligned = [_MockTrajectory(outcome_score=0.9)]
    pairs = build_contrastive_pairs(aligned, [], score_threshold=0.6)
    assert pairs == []


def test_build_contrastive_pairs_no_qualifying():
    aligned = [_MockTrajectory(outcome_score=0.3)]  # below threshold
    misaligned = [_MockTrajectory(outcome_score=0.8)]  # above threshold
    pairs = build_contrastive_pairs(aligned, misaligned, score_threshold=0.6)
    assert pairs == []


def test_build_contrastive_pairs_threshold():
    aligned = [_MockTrajectory(outcome_score=0.6)]  # exactly at threshold
    misaligned = [_MockTrajectory(outcome_score=0.5)]  # just below threshold
    pairs = build_contrastive_pairs(aligned, misaligned, score_threshold=0.6)
    assert len(pairs) == 1


# --- torch-dependent tests ---


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


requires_torch = pytest.mark.skipif(
    not _torch_available(), reason="torch not installed",
)


@requires_torch
def test_contrastive_trajectory_loss_zero_when_separated():
    import torch
    from src.training.cispo import contrastive_trajectory_loss

    pos = torch.tensor([1.0, 0.9])
    neg = torch.tensor([0.1, 0.2])
    loss = contrastive_trajectory_loss(pos, neg, margin=0.5)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


@requires_torch
def test_contrastive_trajectory_loss_positive_when_close():
    import torch
    from src.training.cispo import contrastive_trajectory_loss

    pos = torch.tensor([0.6])
    neg = torch.tensor([0.5])
    # margin=0.5, diff=0.1, loss = max(0, 0.5 - 0.1) = 0.4
    loss = contrastive_trajectory_loss(pos, neg, margin=0.5)
    assert loss.item() == pytest.approx(0.4, abs=1e-6)


@requires_torch
def test_contrastive_trajectory_loss_empty():
    import torch
    from src.training.cispo import contrastive_trajectory_loss

    pos = torch.tensor([])
    neg = torch.tensor([0.5])
    loss = contrastive_trajectory_loss(pos, neg, margin=0.5)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


@requires_torch
def test_contrastive_loss_gradient_flows():
    import torch
    from src.training.cispo import contrastive_trajectory_loss

    pos = torch.tensor([0.6], requires_grad=True)
    neg = torch.tensor([0.5], requires_grad=True)
    loss = contrastive_trajectory_loss(pos, neg, margin=0.5)
    loss.backward()
    assert pos.grad is not None


@requires_torch
def test_infonce_loss_correct_shape():
    import torch
    from src.training.cispo import infonce_trajectory_loss

    pos = torch.randn(3, 16)
    neg = torch.randn(5, 16)
    loss = infonce_trajectory_loss(pos, neg)
    assert loss.dim() == 0  # scalar


@requires_torch
def test_multi_loss_weighted():
    import torch
    from src.training.grpo import multi_loss

    losses = {
        "rl": torch.tensor(1.0),
        "contrastive": torch.tensor(2.0),
    }
    weights = {"rl": 0.7, "contrastive": 0.3}
    result = multi_loss(losses, weights)
    assert result.item() == pytest.approx(0.7 * 1.0 + 0.3 * 2.0, abs=1e-6)


@requires_torch
def test_multi_loss_ignores_zero_weight():
    import torch
    from src.training.grpo import multi_loss

    losses = {"rl": torch.tensor(1.0), "unused": torch.tensor(999.0)}
    weights = {"rl": 1.0, "unused": 0.0}
    result = multi_loss(losses, weights)
    assert result.item() == pytest.approx(1.0, abs=1e-6)
