"""Full NATS event capture to JSONL for audit trail."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.events.bus import EventBus
from src.events.topics import (
    ALIGNMENT_AUDIT,
    ALIGNMENT_EVALS,
    FEEDBACK_SCORED,
    META_ROLLOUTS,
    MODEL_UPDATES,
    RESULTS_CODING,
    SESSIONS,
    SESSION_END,
    SESSION_START,
    SKILLS_CREATED,
    TASKS_CODING,
    TRAINING_COMBINED,
    TRAINING_ROLLOUTS,
)
from src.events.types import AuditLogEvent

logger = logging.getLogger(__name__)

# All known topics to subscribe to
ALL_AUDIT_TOPICS = [
    TASKS_CODING,
    RESULTS_CODING,
    FEEDBACK_SCORED,
    TRAINING_ROLLOUTS,
    TRAINING_COMBINED,
    MODEL_UPDATES,
    SESSIONS,
    SESSION_START,
    SESSION_END,
    META_ROLLOUTS,
    SKILLS_CREATED,
    ALIGNMENT_EVALS,
]


class AuditLogger:
    """Subscribes to all known topics and writes every event to JSONL."""

    def __init__(
        self,
        bus: EventBus,
        log_dir: str = "audit_logs",
        publish_audit_events: bool = True,
    ) -> None:
        self._bus = bus
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._publish_audit_events = publish_audit_events
        self._event_count = 0

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self._log_file = self._log_dir / f"audit_{timestamp}.jsonl"

    async def start(self, topics: list[str] | None = None) -> None:
        """Subscribe to all topics for raw event capture."""
        target_topics = topics or ALL_AUDIT_TOPICS
        for topic in target_topics:
            await self._bus.subscribe_raw(topic, self._handle_raw)
        logger.info("AuditLogger started, monitoring %d topics", len(target_topics))

    async def _handle_raw(self, topic: str, data: bytes) -> None:
        """Handle a raw message from any topic."""
        self._event_count += 1
        payload_str = data.decode("utf-8", errors="replace")

        # Determine event type from payload
        event_type = "unknown"
        try:
            parsed = json.loads(payload_str)
            # Infer type from field presence
            if "task_id" in parsed and "prompt" in parsed and "result" not in parsed:
                event_type = "TaskEvent"
            elif "result" in parsed and "worker_id" in parsed:
                event_type = "ResultEvent"
            elif "score" in parsed and "textual_feedback" in parsed:
                event_type = "FeedbackEvent"
            elif "step_scores" in parsed:
                event_type = "TrainingRolloutEvent"
            elif "model_version" in parsed:
                event_type = "ModelUpdateEvent"
            elif "eval_id" in parsed:
                event_type = "AlignmentEvalEvent"
            elif "skill_name" in parsed:
                event_type = "SkillCreatedEvent"
            elif "session_id" in parsed:
                event_type = "SessionEvent"
        except json.JSONDecodeError:
            pass

        # Write to JSONL
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "event_type": event_type,
            "payload": payload_str,
            "sequence": self._event_count,
        }
        with open(self._log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Publish audit event
        if self._publish_audit_events:
            audit_event = AuditLogEvent(
                original_topic=topic,
                original_event_type=event_type,
                payload_json=payload_str,
                source_component="audit_logger",
            )
            try:
                await self._bus.publish(ALIGNMENT_AUDIT, audit_event)
            except Exception:
                logger.debug("Failed to publish audit event (bus may be closing)")

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def log_file(self) -> Path:
        return self._log_file
