"""Tests for CombinedTrainer — combined RL + OPD loss."""

from __future__ import annotations

import pytest

from src.events.types import CombinedRolloutEvent
from src.training.combined_trainer import CombinedTrainer, CombinedTrainStepResult


try:
    import torch as _torch  # noqa: F401
    _has_torch = True
except ImportError:
    _has_torch = False

requires_torch = pytest.mark.skipif(not _has_torch, reason="torch not installed")


class TestCombinedTrainStepResult:
    def test_serialization(self):
        result = CombinedTrainStepResult(
            loss=0.5, rl_loss=0.3, opd_loss=0.2,
            mean_advantage=0.1, std_advantage=0.05,
            step_count=1, opd_count=2,
        )
        data = result.model_dump()
        assert data["rl_loss"] == 0.3
        assert data["opd_loss"] == 0.2
        assert data["opd_count"] == 2


class TestCombinedRolloutEvent:
    def test_pure_rl_rollout(self):
        event = CombinedRolloutEvent(
            task_id="t1", worker_id="w1", prompt="p", response="r",
            outcome_score=0.8, has_opd=False,
        )
        assert event.has_opd is False
        assert event.hint_text == ""

    def test_opd_rollout(self):
        event = CombinedRolloutEvent(
            task_id="t1", worker_id="w1", prompt="p", response="r",
            outcome_score=0.8, has_opd=True,
            hint_text="use DP", teacher_logprobs=[-0.5, -0.3],
        )
        assert event.has_opd is True
        assert event.hint_text == "use DP"


@requires_torch
@pytest.mark.slow
class TestCombinedTrainerIntegration:
    async def test_pure_rl_batch(self):
        trainer = CombinedTrainer(
            model_name="distilgpt2", device="cpu",
            checkpoint_every=999,
        )
        batch = [
            CombinedRolloutEvent(
                task_id=f"t{i}", worker_id="w1",
                prompt="What is 2+2?", response=f"Answer {i}",
                outcome_score=0.5 + i * 0.1, has_opd=False,
            )
            for i in range(4)
        ]
        result = await trainer.train_step(batch)
        assert isinstance(result, CombinedTrainStepResult)
        assert result.step_count == 1
        assert result.opd_count == 0
        assert result.opd_loss == 0.0

    async def test_mixed_batch(self):
        trainer = CombinedTrainer(
            model_name="distilgpt2", device="cpu",
            checkpoint_every=999,
        )
        batch = [
            CombinedRolloutEvent(
                task_id="t1", worker_id="w1",
                prompt="Q?", response="A1",
                outcome_score=0.8, has_opd=True,
                hint_text="better answer",
            ),
            CombinedRolloutEvent(
                task_id="t2", worker_id="w1",
                prompt="Q?", response="A2",
                outcome_score=0.3, has_opd=False,
            ),
        ]
        result = await trainer.train_step(batch)
        assert result.opd_count == 1
        assert result.rl_loss != 0.0
