"""Tests for the OpenRLHF training backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.events.types import TrainingRolloutEvent
from src.training.openrlhf_backend import OpenRLHFBackend


def _make_rollout(
    prompt: str = "Write a function",
    response: str = "def foo(): pass",
    score: float = 0.8,
    task_id: str = "test-task-1",
    worker_id: str = "worker-01",
) -> TrainingRolloutEvent:
    return TrainingRolloutEvent(
        task_id=task_id,
        worker_id=worker_id,
        prompt=prompt,
        response=response,
        steps=["step 1", "step 2"],
        step_scores=[0.7, 0.9],
        outcome_score=score,
    )


class TestOpenRLHFBackend:
    def test_implements_trainer_protocol(self):
        backend = OpenRLHFBackend()
        assert hasattr(backend, "train_step")
        assert hasattr(backend, "checkpoint_path")

    def test_export_rollouts(self, tmp_path: Path):
        backend = OpenRLHFBackend(dataset_dir=str(tmp_path / "datasets"))
        rollouts = [_make_rollout(), _make_rollout(score=0.6, task_id="test-task-2")]
        path = tmp_path / "datasets" / "batch.jsonl"
        count = backend._export_rollouts(rollouts, path)

        assert count == 2
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

        record = json.loads(lines[0])
        assert record["prompt"] == "Write a function"
        assert record["reward"] == 0.8
        assert record["step_scores"] == [0.7, 0.9]

    @pytest.mark.asyncio
    async def test_train_step_simulation(self, tmp_path: Path):
        backend = OpenRLHFBackend(
            output_dir=str(tmp_path / "checkpoints"),
            dataset_dir=str(tmp_path / "datasets"),
        )
        batch = [_make_rollout(), _make_rollout(score=0.6)]

        result = await backend.train_step(batch)

        assert result.step_count == 1
        assert result.loss > 0
        assert result.checkpoint_path is not None
        assert Path(result.checkpoint_path).exists()

        # Verify metadata file was created
        meta_path = Path(result.checkpoint_path) / "training_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["batch_size"] == 2
        assert meta["backend"] == "openrlhf_simulation"

    @pytest.mark.asyncio
    async def test_train_step_exports_dataset(self, tmp_path: Path):
        backend = OpenRLHFBackend(
            output_dir=str(tmp_path / "checkpoints"),
            dataset_dir=str(tmp_path / "datasets"),
        )
        batch = [_make_rollout()]

        await backend.train_step(batch)

        dataset_files = list((tmp_path / "datasets").glob("*.jsonl"))
        assert len(dataset_files) == 1

    @pytest.mark.asyncio
    async def test_step_count_increments(self, tmp_path: Path):
        backend = OpenRLHFBackend(
            output_dir=str(tmp_path / "checkpoints"),
            dataset_dir=str(tmp_path / "datasets"),
        )
        batch = [_make_rollout()]

        r1 = await backend.train_step(batch)
        r2 = await backend.train_step(batch)

        assert r1.step_count == 1
        assert r2.step_count == 2

    def test_checkpoint_path_initially_none(self):
        backend = OpenRLHFBackend()
        assert backend.checkpoint_path() is None

    @pytest.mark.asyncio
    async def test_checkpoint_path_after_training(self, tmp_path: Path):
        backend = OpenRLHFBackend(
            output_dir=str(tmp_path / "checkpoints"),
            dataset_dir=str(tmp_path / "datasets"),
        )
        await backend.train_step([_make_rollout()])
        assert backend.checkpoint_path() is not None

    def test_parse_loss_from_output(self):
        output = "step 1: loss 0.5432\nstep 2: loss 0.3210\n"
        loss = OpenRLHFBackend._parse_loss_from_output(output)
        assert loss == pytest.approx(0.3210)

    def test_parse_loss_empty_output(self):
        loss = OpenRLHFBackend._parse_loss_from_output("")
        assert loss == 0.0
