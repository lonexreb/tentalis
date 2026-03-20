"""SkillEvolver — subscribes to feedback, extracts corrective skills via LLM."""

from __future__ import annotations

import json
import logging
import uuid

from src.events.bus import EventBus
from src.events.topics import FEEDBACK_SCORED, SKILLS_CREATED
from src.events.types import FeedbackEvent, SkillCreatedEvent
from src.inference.client import InferenceClient
from src.skills.retriever import SkillRetriever
from src.skills.store import Skill, SkillStore

logger = logging.getLogger(__name__)

SKILL_EXTRACTION_PROMPT = """\
A worker received a low score ({score:.2f}) on a task. The feedback was:

"{feedback}"

Extract a reusable corrective skill that would help avoid this mistake in the future.

Respond in JSON:
{{
  "skill_name": "short descriptive name",
  "skill_text": "concise instruction the worker should follow",
  "category": "one of: reasoning, formatting, accuracy, completeness, general"
}}"""


class SkillEvolver:
    """Watches feedback events and creates skills from low-scoring results."""

    def __init__(
        self,
        bus: EventBus,
        store: SkillStore,
        retriever: SkillRetriever,
        client: InferenceClient,
        *,
        model: str = "qwen2.5:1.5b",
        threshold: float = 0.4,
    ) -> None:
        self._bus = bus
        self._store = store
        self._retriever = retriever
        self._client = client
        self._model = model
        self._threshold = threshold

    async def start(self) -> None:
        await self._bus.subscribe(
            FEEDBACK_SCORED,
            FeedbackEvent,
            self._handle_feedback,
        )
        logger.info(
            "SkillEvolver started (threshold=%.2f, model=%s)",
            self._threshold,
            self._model,
        )

    async def _handle_feedback(self, event: FeedbackEvent) -> None:
        if event.score >= self._threshold:
            return
        if not event.textual_feedback:
            return

        logger.info(
            "Low score %.2f for task %s — extracting skill",
            event.score,
            event.task_id,
        )

        try:
            skill = await self._extract_skill(event)
            if skill:
                self._store.add(skill)
                await self._publish_skill_created(event, skill)
        except Exception:
            logger.exception("Failed to extract skill from feedback %s", event.task_id)

    async def _extract_skill(self, event: FeedbackEvent) -> Skill | None:
        prompt = SKILL_EXTRACTION_PROMPT.format(
            score=event.score,
            feedback=event.textual_feedback,
        )
        raw = await self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            json_mode=True,
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse skill extraction response")
            return None

        skill_name = parsed.get("skill_name", "unnamed skill")
        skill_text = parsed.get("skill_text", "")
        category = parsed.get("category", "general")

        if not skill_text:
            return None

        embedding = self._retriever.encode(skill_text)

        return Skill(
            skill_id=str(uuid.uuid4()),
            name=skill_name,
            text=skill_text,
            category=category,
            embedding=embedding,
            source_task_id=event.task_id,
            source_score=event.score,
        )

    async def _publish_skill_created(
        self, event: FeedbackEvent, skill: Skill
    ) -> None:
        skill_event = SkillCreatedEvent(
            skill_id=skill.skill_id,
            task_id=event.task_id,
            worker_id=event.worker_id,
            skill_name=skill.name,
            skill_text=skill.text,
            category=skill.category,
            source_feedback=event.textual_feedback,
            source_score=event.score,
        )
        await self._bus.publish(SKILLS_CREATED, skill_event)
        logger.info("Published SkillCreatedEvent: %s", skill.name)
