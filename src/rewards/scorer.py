from __future__ import annotations

import json
import logging
from typing import Protocol, runtime_checkable

from src.inference.client import InferenceClient
from src.rewards.prompts import STEP_JUDGE_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_SCORE = 0.5


@runtime_checkable
class StepScorer(Protocol):
    async def score_steps(self, prompt: str, steps: list[str]) -> list[float]: ...


class LLMJudgeScorer:
    def __init__(
        self,
        *,
        model: str = "qwen2.5:1.5b",
        client: InferenceClient,
        num_votes: int = 1,
    ) -> None:
        self.model = model
        self._client = client
        self._num_votes = max(1, num_votes)

    async def score_steps(self, prompt: str, steps: list[str]) -> list[float]:
        scores: list[float] = []
        for i, step in enumerate(steps, 1):
            formatted_steps = "\n".join(
                f"  {j}. {s}" for j, s in enumerate(steps[:i], 1)
            )
            judge_prompt = STEP_JUDGE_PROMPT.format(
                step_num=i,
                task_description=prompt,
                formatted_steps=formatted_steps,
                current_step=step,
            )
            if self._num_votes <= 1:
                score = await self._judge_single_step(judge_prompt)
            else:
                score = await self._judge_with_voting(judge_prompt)
            scores.append(score)
        return scores

    async def _judge_with_voting(self, judge_prompt: str) -> float:
        """Run parallel evaluations and return the median score."""
        import asyncio

        tasks = [
            self._judge_single_step(judge_prompt)
            for _ in range(self._num_votes)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_scores = [
            r for r in results
            if isinstance(r, float)
        ]
        if not valid_scores:
            logger.warning("All %d votes failed, using fallback", self._num_votes)
            return DEFAULT_FALLBACK_SCORE
        valid_scores.sort()
        mid = len(valid_scores) // 2
        if len(valid_scores) % 2 == 0:
            return (valid_scores[mid - 1] + valid_scores[mid]) / 2.0
        return valid_scores[mid]

    async def _judge_single_step(self, judge_prompt: str) -> float:
        try:
            raw = await self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": judge_prompt}],
                json_mode=True,
            )
            parsed = json.loads(raw)
            progress = float(parsed.get("progress", DEFAULT_FALLBACK_SCORE))
            correctness = float(parsed.get("correctness", DEFAULT_FALLBACK_SCORE))
            return (progress + correctness) / 2.0
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse judge response: %s", exc)
            return DEFAULT_FALLBACK_SCORE
