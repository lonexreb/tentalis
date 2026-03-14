"""SWEWorker — GitHub issue → code patch with plan/implement/test steps."""

from __future__ import annotations

import logging

from src.events.bus import EventBus
from src.events.types import ResultEvent, TaskEvent, TaskStatus
from src.inference.client import InferenceClient
from src.workers.base import BaseWorker

logger = logging.getLogger(__name__)

PLAN_PROMPT = """\
You are a software engineer. Given the following issue, create a step-by-step plan \
to fix it. Output ONLY the plan as numbered steps.

Issue: {issue}"""

IMPLEMENT_PROMPT = """\
Implement the following plan as a unified diff patch.

Plan:
{plan}

Original issue: {issue}

Output ONLY the diff patch."""

TEST_PROMPT = """\
Write test cases that verify the fix described below.

Issue: {issue}
Fix plan: {plan}

Output ONLY the test code."""


class SWEWorker(BaseWorker):
    """Solves GitHub issues via a plan → implement → test pipeline.

    Steps: ["plan: ...", "implement: ...", "test: ..."]
    """

    environment_type = "swe"

    def __init__(
        self,
        worker_id: str,
        bus: EventBus,
        client: InferenceClient,
        model: str = "qwen2.5:1.5b",
        task_types: list[str] | None = None,
    ) -> None:
        super().__init__(worker_id, bus, task_types or ["swe"])
        self._client = client
        self._model = model

    async def process(self, task: TaskEvent) -> ResultEvent:
        issue = task.prompt
        steps: list[str] = []

        try:
            # Step 1: Plan
            plan = await self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": PLAN_PROMPT.format(issue=issue)}],
            )
            steps.append(f"plan: {plan}")

            # Step 2: Implement
            patch = await self._client.chat(
                model=self._model,
                messages=[
                    {"role": "user", "content": IMPLEMENT_PROMPT.format(
                        plan=plan, issue=issue,
                    )},
                ],
            )
            steps.append(f"implement: {patch}")

            # Step 3: Test
            tests = await self._client.chat(
                model=self._model,
                messages=[
                    {"role": "user", "content": TEST_PROMPT.format(
                        issue=issue, plan=plan,
                    )},
                ],
            )
            steps.append(f"test: {tests}")

            return ResultEvent(
                task_id=task.task_id,
                worker_id=self.worker_id,
                result=patch,
                status=TaskStatus.SUCCESS,
                steps=steps,
            )
        except Exception as exc:
            logger.exception("SWEWorker failed on task %s", task.task_id)
            steps.append(f"error: {exc}")
            return ResultEvent(
                task_id=task.task_id,
                worker_id=self.worker_id,
                result=str(exc),
                status=TaskStatus.FAILED,
                steps=steps,
            )
