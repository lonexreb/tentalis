"""Tests for Trainer protocol, MockTrainer, and GRPOTrainer."""

import pytest

from src.events.types import TrainingRolloutEvent
from src.training.trainer import MockTrainer, Trainer, TrainStepResult


def _make_rollout(
    prompt: str = "Write fibonacci",
    outcome_score: float = 0.8,
) -> TrainingRolloutEvent:
    return TrainingRolloutEvent(
        task_id="t-1",
        worker_id="w-1",
        prompt=prompt,
        response="def fib(n): ...",
        steps=["think", "code"],
        step_scores=[0.7, 0.9],
        outcome_score=outcome_score,
    )


class TestTrainStepResult:
    def test_serialization(self):
        result = TrainStepResult(
            loss=0.5, mean_advantage=0.1, std_advantage=0.3,
            checkpoint_path="/tmp/v0001", step_count=1,
        )
        data = result.model_dump_json()
        restored = TrainStepResult.model_validate_json(data)
        assert restored.loss == 0.5
        assert restored.checkpoint_path == "/tmp/v0001"

    def test_defaults(self):
        result = TrainStepResult(loss=0.1, mean_advantage=0.0, std_advantage=1.0)
        assert result.checkpoint_path is None
        assert result.step_count == 0


class TestMockTrainer:
    def test_protocol_conformance(self):
        assert isinstance(MockTrainer(), Trainer)

    @pytest.mark.asyncio
    async def test_train_step_returns_result(self):
        trainer = MockTrainer()
        batch = [_make_rollout() for _ in range(4)]
        result = await trainer.train_step(batch)
        assert isinstance(result, TrainStepResult)
        assert result.step_count == 1
        assert result.loss == 1.0

    @pytest.mark.asyncio
    async def test_step_count_increments(self):
        trainer = MockTrainer()
        batch = [_make_rollout()]
        await trainer.train_step(batch)
        result = await trainer.train_step(batch)
        assert result.step_count == 2
        assert result.loss == 0.5  # 1.0 / step_count

    def test_checkpoint_path_is_none(self):
        trainer = MockTrainer()
        assert trainer.checkpoint_path() is None


# GRPOTrainer tests — require torch, transformers, peft
try:
    import torch
    import transformers
    import peft
    _has_training_deps = True
except ImportError:
    _has_training_deps = False

requires_training = pytest.mark.skipif(
    not _has_training_deps, reason="torch/transformers/peft not installed"
)


@requires_training
class TestGRPOTrainer:
    @pytest.fixture
    def trainer(self, tmp_path):
        from src.training.trainer import GRPOTrainer

        return GRPOTrainer(
            model_name="distilgpt2",
            lr=1e-4,
            checkpoint_dir=str(tmp_path / "ckpts"),
            lora_rank=4,
            checkpoint_every=1,
            device="cpu",
        )

    def test_protocol_conformance(self, trainer):
        assert isinstance(trainer, Trainer)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_train_step(self, trainer):
        batch = [
            _make_rollout(prompt="Write fibonacci", outcome_score=0.9),
            _make_rollout(prompt="Write fibonacci", outcome_score=0.3),
            _make_rollout(prompt="Write hello world", outcome_score=0.7),
            _make_rollout(prompt="Write hello world", outcome_score=0.5),
        ]
        result = await trainer.train_step(batch)
        assert isinstance(result, TrainStepResult)
        assert result.step_count == 1
        assert result.loss != 0.0
        # checkpoint_every=1, so checkpoint should be saved
        assert result.checkpoint_path is not None

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_checkpoint_saved(self, trainer, tmp_path):
        batch = [_make_rollout() for _ in range(2)]
        result = await trainer.train_step(batch)
        assert result.checkpoint_path is not None
        ckpt = tmp_path / "ckpts" / "v0001"
        assert ckpt.exists()
        assert trainer.checkpoint_path() == str(ckpt)
