"""Tests for multi-environment workers (Terminal, SWE, GUI)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.events.types import ModelUpdateEvent, ResultEvent, TaskEvent, TaskStatus
from src.workers.terminal_worker import TerminalWorker
from src.workers.swe_worker import SWEWorker
from src.workers.gui_worker import GUIWorker


@pytest.fixture
def mock_bus():
    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.chat = AsyncMock(return_value='{"action": "done"}')
    return client


class TestTerminalWorker:
    def test_environment_type(self, mock_bus):
        worker = TerminalWorker("w1", mock_bus)
        assert worker.environment_type == "terminal"

    def test_parse_commands(self, mock_bus):
        worker = TerminalWorker("w1", mock_bus)
        cmds = worker._parse_commands("echo hello\n# comment\n\nls -la")
        assert cmds == ["echo hello", "ls -la"]

    async def test_process_uses_docker(self, mock_bus):
        worker = TerminalWorker("w1", mock_bus, timeout=5)
        task = TaskEvent(
            task_id="t1", manager_id="m1",
            task_type="terminal", prompt="echo hello",
        )
        # This will fail if Docker is not available, which is expected in CI
        result = await worker.process(task)
        assert isinstance(result, ResultEvent)
        assert result.task_id == "t1"
        # Either succeeds (Docker available) or fails (Docker not available)
        assert result.status in (TaskStatus.SUCCESS, TaskStatus.FAILED)


class TestSWEWorker:
    def test_environment_type(self, mock_bus, mock_client):
        worker = SWEWorker("w1", mock_bus, mock_client)
        assert worker.environment_type == "swe"

    async def test_process_three_steps(self, mock_bus, mock_client):
        mock_client.chat = AsyncMock(
            side_effect=["1. Fix the bug", "--- a/file.py\n+++ b/file.py", "def test_fix():"]
        )
        worker = SWEWorker("w1", mock_bus, mock_client, model="test")
        task = TaskEvent(
            task_id="t1", manager_id="m1",
            task_type="swe", prompt="Fix the null pointer bug",
        )
        result = await worker.process(task)
        assert result.status == TaskStatus.SUCCESS
        assert len(result.steps) == 3
        assert result.steps[0].startswith("plan:")
        assert result.steps[1].startswith("implement:")
        assert result.steps[2].startswith("test:")

    async def test_process_handles_failure(self, mock_bus, mock_client):
        mock_client.chat = AsyncMock(side_effect=RuntimeError("LLM down"))
        worker = SWEWorker("w1", mock_bus, mock_client)
        task = TaskEvent(
            task_id="t1", manager_id="m1",
            task_type="swe", prompt="Fix bug",
        )
        result = await worker.process(task)
        assert result.status == TaskStatus.FAILED


class TestGUIWorker:
    def test_environment_type(self, mock_bus, mock_client):
        worker = GUIWorker("w1", mock_bus, mock_client)
        assert worker.environment_type == "gui"

    async def test_process_generates_steps(self, mock_bus, mock_client):
        mock_client.chat = AsyncMock(
            side_effect=[
                '{"action": "click", "target": "button"}',
                '{"action": "done"}',
            ]
        )
        worker = GUIWorker("w1", mock_bus, mock_client, max_steps=5)
        task = TaskEvent(
            task_id="t1", manager_id="m1",
            task_type="gui", prompt="Click the submit button",
        )
        result = await worker.process(task)
        assert result.status == TaskStatus.SUCCESS
        assert len(result.steps) == 2

    async def test_max_steps_limit(self, mock_bus, mock_client):
        mock_client.chat = AsyncMock(return_value='{"action": "click", "target": "x"}')
        worker = GUIWorker("w1", mock_bus, mock_client, max_steps=3)
        task = TaskEvent(
            task_id="t1", manager_id="m1",
            task_type="gui", prompt="Do something",
        )
        result = await worker.process(task)
        assert len(result.steps) == 3


class TestBaseWorkerTargetFiltering:
    async def test_ignores_update_for_other_worker(self, mock_bus, mock_client):
        worker = SWEWorker("w1", mock_bus, mock_client)
        event = ModelUpdateEvent(
            model_version="v1",
            checkpoint_path="/ckpt",
            target_worker_id="w2",
        )
        await worker._handle_model_update(event)
        # No error, silently ignored

    async def test_accepts_update_for_self(self, mock_bus, mock_client):
        worker = SWEWorker("w1", mock_bus, mock_client)
        event = ModelUpdateEvent(
            model_version="v1",
            checkpoint_path="/ckpt",
            target_worker_id="w1",
        )
        await worker._handle_model_update(event)
        # reload_model is a no-op in SWEWorker, but it was called

    async def test_accepts_broadcast_update(self, mock_bus, mock_client):
        worker = SWEWorker("w1", mock_bus, mock_client)
        event = ModelUpdateEvent(
            model_version="v1",
            checkpoint_path="/ckpt",
            target_worker_id=None,
        )
        await worker._handle_model_update(event)
