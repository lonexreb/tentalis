"""Tests for DAPO — dynamic sampling, entropy bonus, dapo loss."""

from __future__ import annotations

import pytest

from src.events.types import TrainingRolloutEvent
from src.training.dapo import dynamic_sample_filter


def _make_rollout(
    prompt: str = "test",
    response: str = "answer",
    outcome_score: float = 0.5,
) -> TrainingRolloutEvent:
    return TrainingRolloutEvent(
        task_id="t1",
        worker_id="w1",
        prompt=prompt,
        response=response,
        steps=["step1"],
        step_scores=[outcome_score],
        outcome_score=outcome_score,
    )


def test_dynamic_filter_removes_low_reward():
    rollouts = [
        _make_rollout(response="low", outcome_score=0.05),
        _make_rollout(response="mid", outcome_score=0.5),
        _make_rollout(response="high", outcome_score=0.9),
    ]
    filtered = dynamic_sample_filter(rollouts, min_reward_threshold=0.1)
    assert len(filtered) == 2
    assert all(r.outcome_score >= 0.1 for r in filtered)


def test_dynamic_filter_removes_duplicates():
    rollouts = [
        _make_rollout(response="same", outcome_score=0.5),
        _make_rollout(response="same", outcome_score=0.6),
        _make_rollout(response="different", outcome_score=0.7),
    ]
    filtered = dynamic_sample_filter(rollouts, min_reward_threshold=0.0)
    assert len(filtered) == 2
    responses = {r.response for r in filtered}
    assert responses == {"same", "different"}


def test_dynamic_filter_preserves_diverse():
    rollouts = [
        _make_rollout(response="a", outcome_score=0.5),
        _make_rollout(response="b", outcome_score=0.6),
        _make_rollout(response="c", outcome_score=0.7),
    ]
    filtered = dynamic_sample_filter(rollouts, min_reward_threshold=0.0)
    assert len(filtered) == 3


def test_dynamic_filter_empty_batch():
    assert dynamic_sample_filter([], min_reward_threshold=0.1) == []


def test_dynamic_filter_all_below_threshold():
    rollouts = [
        _make_rollout(outcome_score=0.01),
        _make_rollout(outcome_score=0.05),
    ]
    filtered = dynamic_sample_filter(rollouts, min_reward_threshold=0.1)
    assert filtered == []


def test_dynamic_filter_groups_by_prompt():
    rollouts = [
        _make_rollout(prompt="p1", response="a", outcome_score=0.5),
        _make_rollout(prompt="p1", response="a", outcome_score=0.6),  # dup in p1
        _make_rollout(prompt="p2", response="a", outcome_score=0.7),  # not dup (diff prompt)
    ]
    filtered = dynamic_sample_filter(rollouts, min_reward_threshold=0.0)
    # p1 group: "a" deduped → 1 rollout. p2 group: "a" → 1 rollout.
    assert len(filtered) == 2


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
def test_entropy_bonus_positive():
    import torch
    from src.training.dapo import entropy_bonus

    logits = torch.randn(2, 10, 50)  # (batch, seq, vocab)
    bonus = entropy_bonus(logits, beta_entropy=0.01)
    assert bonus.item() > 0


@requires_torch
def test_entropy_bonus_near_zero_for_peaked():
    import torch
    from src.training.dapo import entropy_bonus

    # Very peaked distribution (one-hot-ish)
    logits = torch.full((2, 5, 100), -100.0)
    logits[:, :, 0] = 100.0  # all mass on token 0
    bonus = entropy_bonus(logits, beta_entropy=0.01)
    assert bonus.item() < 1e-4


@requires_torch
def test_dapo_loss_includes_entropy():
    import torch
    from src.training.dapo import dapo_loss
    from src.training.grpo import asymmetric_clipped_surrogate_loss

    ratios = torch.ones(4)
    advantages = torch.tensor([0.5, -0.5, 0.3, -0.3])
    logits = torch.randn(4, 10, 50)

    dapo = dapo_loss(ratios, advantages, logits, beta_entropy=0.01)
    clip_only = asymmetric_clipped_surrogate_loss(ratios, advantages)
    # DAPO loss should differ from pure clip loss by the entropy bonus
    assert not torch.isclose(dapo, clip_only)


@requires_torch
def test_dapo_loss_gradient_flows():
    import torch
    from src.training.dapo import dapo_loss

    ratios = torch.ones(4, requires_grad=True)
    advantages = torch.tensor([0.5, -0.5, 0.3, -0.3])
    logits = torch.randn(4, 10, 50, requires_grad=True)

    loss = dapo_loss(ratios, advantages, logits, beta_entropy=0.01)
    loss.backward()
    assert logits.grad is not None
