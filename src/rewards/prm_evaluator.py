from __future__ import annotations

import logging

from src.events.bus import EventBus
from src.events.topics import TRAINING_ROLLOUTS, result_topic
from src.events.types import ResultEvent, TaskStatus, TrainingRolloutEvent
from src.rewards.scorer import StepScorer

logger = logging.getLogger(__name__)


class PRMEvaluator:
    def __init__(self, bus: EventBus, scorer: StepScorer) -> None:
        self.bus = bus
        self.scorer = scorer

    async def start(self, task_types: list[str]) -> None:
        for tt in task_types:
            await self.bus.subscribe(
                result_topic(tt), ResultEvent, self._handle_result
            )
        logger.info("PRMEvaluator started, listening for %s", task_types)

    async def _handle_result(self, result: ResultEvent) -> None:
        if result.status != TaskStatus.SUCCESS:
            logger.debug("Skipping non-success result %s", result.task_id)
            return

        if not result.steps:
            logger.debug("No steps to score for %s", result.task_id)
            return

        # Support CombinedScorer's environment_type kwarg
        from src.rewards.combined_scorer import CombinedScorer

        if isinstance(self.scorer, CombinedScorer):
            step_scores = await self.scorer.score_steps(
                result.prompt, result.steps,
                environment_type=result.model_dump().get("metadata", {}).get(
                    "environment_type", "chat"
                ),
            )
        else:
            step_scores = await self.scorer.score_steps(result.prompt, result.steps)

        outcome_score = step_scores[-1] if step_scores else 0.0

        rollout = TrainingRolloutEvent(
            task_id=result.task_id,
            worker_id=result.worker_id,
            prompt=result.prompt,
            response=result.result,
            steps=result.steps,
            step_scores=step_scores,
            outcome_score=max(0.0, min(1.0, outcome_score)),
        )
        await self.bus.publish(TRAINING_ROLLOUTS, rollout)
        logger.info(
            "Published rollout for %s — step_scores=%s outcome=%.3f",
            result.task_id,
            step_scores,
            outcome_score,
        )
