"""Tests for audit logger."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.alignment.audit_logger import AuditLogger


@pytest.fixture
def mock_bus():
    bus = AsyncMock()
    bus.subscribe_raw = AsyncMock()
    bus.publish = AsyncMock()
    return bus


class TestAuditLogger:
    async def test_start_subscribes_to_topics(self, mock_bus, tmp_path):
        logger = AuditLogger(mock_bus, log_dir=str(tmp_path))
        await logger.start()
        assert mock_bus.subscribe_raw.call_count > 0

    async def test_start_with_custom_topics(self, mock_bus, tmp_path):
        logger = AuditLogger(mock_bus, log_dir=str(tmp_path))
        await logger.start(topics=["test.topic1", "test.topic2"])
        assert mock_bus.subscribe_raw.call_count == 2

    async def test_handle_raw_writes_jsonl(self, mock_bus, tmp_path):
        logger = AuditLogger(mock_bus, log_dir=str(tmp_path), publish_audit_events=False)
        payload = json.dumps({"task_id": "t1", "prompt": "hello"}).encode()
        await logger._handle_raw("tasks.coding", payload)

        assert logger.event_count == 1
        content = logger.log_file.read_text().strip()
        entry = json.loads(content)
        assert entry["topic"] == "tasks.coding"
        assert entry["event_type"] == "TaskEvent"
        assert entry["sequence"] == 1

    async def test_handle_raw_detects_result_event(self, mock_bus, tmp_path):
        logger = AuditLogger(mock_bus, log_dir=str(tmp_path), publish_audit_events=False)
        payload = json.dumps({
            "task_id": "t1",
            "worker_id": "w1",
            "result": "answer",
        }).encode()
        await logger._handle_raw("results.coding", payload)

        content = logger.log_file.read_text().strip()
        entry = json.loads(content)
        assert entry["event_type"] == "ResultEvent"

    async def test_handle_raw_detects_feedback_event(self, mock_bus, tmp_path):
        logger = AuditLogger(mock_bus, log_dir=str(tmp_path), publish_audit_events=False)
        payload = json.dumps({
            "task_id": "t1",
            "score": 0.8,
            "textual_feedback": "good",
        }).encode()
        await logger._handle_raw("feedback.scored", payload)

        content = logger.log_file.read_text().strip()
        entry = json.loads(content)
        assert entry["event_type"] == "FeedbackEvent"

    async def test_publishes_audit_event(self, mock_bus, tmp_path):
        logger = AuditLogger(mock_bus, log_dir=str(tmp_path), publish_audit_events=True)
        payload = json.dumps({"task_id": "t1", "prompt": "hello"}).encode()
        await logger._handle_raw("tasks.coding", payload)

        mock_bus.publish.assert_called_once()
        call_args = mock_bus.publish.call_args
        assert call_args[0][0] == "alignment.audit"

    async def test_multiple_events_sequential(self, mock_bus, tmp_path):
        logger = AuditLogger(mock_bus, log_dir=str(tmp_path), publish_audit_events=False)
        for i in range(5):
            payload = json.dumps({"task_id": f"t{i}", "prompt": f"p{i}"}).encode()
            await logger._handle_raw("tasks.coding", payload)

        assert logger.event_count == 5
        lines = logger.log_file.read_text().strip().split("\n")
        assert len(lines) == 5

    async def test_handles_invalid_json(self, mock_bus, tmp_path):
        logger = AuditLogger(mock_bus, log_dir=str(tmp_path), publish_audit_events=False)
        await logger._handle_raw("test.topic", b"not json at all")

        assert logger.event_count == 1
        content = logger.log_file.read_text().strip()
        entry = json.loads(content)
        assert entry["event_type"] == "unknown"

    async def test_log_file_property(self, mock_bus, tmp_path):
        logger = AuditLogger(mock_bus, log_dir=str(tmp_path))
        assert logger.log_file.parent == tmp_path
        assert logger.log_file.suffix == ".jsonl"
