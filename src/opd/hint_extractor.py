"""HintExtractor — converts manager textual feedback into OPD training signals."""

from __future__ import annotations

import logging
from collections import OrderedDict

from src.events.bus import EventBus
from src.events.topics import FEEDBACK_SCORED, OPD_HINTS
from src.events.types import FeedbackEvent, OPDHintEvent, ResultEvent
from src.inference.client import InferenceClient

logger = logging.getLogger(__name__)

HINT_EXTRACT_PROMPT = """\
A worker produced a response to a task, and a manager gave the following feedback:

"{feedback}"

Extract the core corrective instruction from this feedback as a concise hint \
that could guide the worker to produce a better response. \
Reply with ONLY the hint text, nothing else."""


class HintExtractor:
    """Subscribes to feedback, extracts OPD hints, queries teacher for logprobs."""

    def __init__(
        self,
        bus: EventBus,
        teacher_client: InferenceClient,
        teacher_model: str = "qwen2.5:1.5b",
        result_cache_size: int = 1000,
    ) -> None:
        self._bus = bus
        self._teacher = teacher_client
        self._teacher_model = teacher_model
        # LRU cache of task_id → ResultEvent for joining
        self._result_cache: OrderedDict[str, ResultEvent] = OrderedDict()
        self._cache_size = result_cache_size

    async def start(self, task_types: list[str]) -> None:
        from src.events.topics import result_topic

        for tt in task_types:
            await self._bus.subscribe(
                result_topic(tt), ResultEvent, self._cache_result
            )
        await self._bus.subscribe(
            FEEDBACK_SCORED, FeedbackEvent, self._handle_feedback
        )
        logger.info("HintExtractor started")

    async def _cache_result(self, result: ResultEvent) -> None:
        self._result_cache[result.task_id] = result
        if len(self._result_cache) > self._cache_size:
            self._result_cache.popitem(last=False)

    async def _handle_feedback(self, feedback: FeedbackEvent) -> None:
        if not feedback.textual_feedback:
            return

        logger.info("Extracting OPD hint for task %s", feedback.task_id)

        # Extract corrective hint from feedback text
        hint_text = await self._extract_hint(feedback.textual_feedback)

        # Get teacher logprobs on the corrected prompt+hint
        cached_result = self._result_cache.get(feedback.task_id)
        teacher_logprobs = await self._get_teacher_logprobs(
            cached_result, hint_text
        )

        event = OPDHintEvent(
            task_id=feedback.task_id,
            worker_id=feedback.worker_id,
            hint_text=hint_text,
            teacher_logprobs=teacher_logprobs,
            original_response=cached_result.result if cached_result else "",
        )
        await self._bus.publish(OPD_HINTS, event)
        logger.info("Published OPDHintEvent for task %s", feedback.task_id)

    async def _extract_hint(self, feedback_text: str) -> str:
        prompt = HINT_EXTRACT_PROMPT.format(feedback=feedback_text)
        try:
            return await self._teacher.chat(
                model=self._teacher_model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:
            logger.warning("Failed to extract hint, using raw feedback", exc_info=True)
            return feedback_text

    async def _get_teacher_logprobs(
        self,
        cached_result: ResultEvent | None,
        hint_text: str,
    ) -> list[float]:
        """Query teacher model with hint to get reference logprobs.

        Requires an OpenAI-compatible backend that returns logprobs.
        Falls back to empty list if unavailable.
        """
        if not cached_result:
            return []

        messages = [
            {"role": "user", "content": cached_result.prompt},
            {"role": "assistant", "content": hint_text},
        ]
        try:
            # Use teacher model to get a response — logprobs come from the
            # intercept proxy or are extracted at the trainer level
            await self._teacher.chat(
                model=self._teacher_model,
                messages=messages,
            )
            # Actual logprobs are extracted by the intercept proxy's SessionEvent
            # and joined at the CombinedRolloutBuilder level
            return []
        except Exception:
            logger.warning("Failed to get teacher logprobs", exc_info=True)
            return []
