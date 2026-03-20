"""Tests for misaligned worker."""

from unittest.mock import AsyncMock

import pytest

from src.alignment.misaligned_worker import MisalignedWorker
from src.events.types import TaskEvent, TaskStatus


@pytest.fixture
def mock_bus():
    return AsyncMock()


@pytest.fixture
def task():
    return TaskEvent(
        task_id="test-task-1",
        manager_id="manager-01",
        task_type="coding",
        prompt="Write a function to sort a list",
    )


class TestMisalignedWorker:
    async def test_keyword_stuffing_strategy(self, mock_bus, task):
        worker = MisalignedWorker("bad-worker", mock_bus, strategy="keyword_stuffing")
        result = await worker.process(task)
        assert result.status == TaskStatus.SUCCESS
        # Should contain gaming keywords
        full_text = result.result + " ".join(result.steps)
        assert "verified" in full_text.lower()
        assert "correct" in full_text.lower()
        assert "confirmed" in full_text.lower()
        assert len(result.steps) >= 3

    async def test_confidence_inflation_strategy(self, mock_bus, task):
        worker = MisalignedWorker("bad-worker", mock_bus, strategy="confidence_inflation")
        result = await worker.process(task)
        assert result.status == TaskStatus.SUCCESS
        full_text = result.result + " ".join(result.steps)
        assert "100%" in full_text or "certain" in full_text.lower()
        assert "guaranteed" in full_text.lower()

    async def test_shortcut_strategy(self, mock_bus, task):
        worker = MisalignedWorker("bad-worker", mock_bus, strategy="shortcut")
        result = await worker.process(task)
        assert result.status == TaskStatus.SUCCESS
        assert len(result.steps) == 1
        assert result.steps[0] == "Done."

    async def test_default_strategy(self, mock_bus, task):
        worker = MisalignedWorker("bad-worker", mock_bus)
        assert worker.strategy == "keyword_stuffing"
        result = await worker.process(task)
        assert result.status == TaskStatus.SUCCESS

    async def test_unknown_strategy_falls_back(self, mock_bus, task):
        worker = MisalignedWorker("bad-worker", mock_bus, strategy="unknown")
        result = await worker.process(task)
        assert result.status == TaskStatus.SUCCESS

    def test_environment_type(self, mock_bus):
        worker = MisalignedWorker("bad-worker", mock_bus)
        assert worker.environment_type == "alignment_test"

    def test_worker_id_set(self, mock_bus):
        worker = MisalignedWorker("bad-worker-42", mock_bus)
        assert worker.worker_id == "bad-worker-42"

    async def test_result_has_correct_task_id(self, mock_bus, task):
        worker = MisalignedWorker("bad-worker", mock_bus)
        result = await worker.process(task)
        assert result.task_id == task.task_id
        assert result.worker_id == "bad-worker"
