from __future__ import annotations

import logging
import re

from src.events.types import ModelUpdateEvent, ResultEvent, TaskEvent, TaskStatus
from src.inference.client import InferenceClient
from src.workers.base import BaseWorker

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a helpful assistant that solves tasks step by step.

Format your response as numbered steps wrapped in <step> tags, followed by a final answer in an <answer> tag.

Example:
<step>1. Understand the problem requirements.</step>
<step>2. Implement the solution.</step>
<answer>Here is the final answer.</answer>

Always use this format. Think carefully through each step."""

_STEP_RE = re.compile(r"<step>(.*?)</step>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _parse_steps(text: str) -> tuple[list[str], str]:
    """Extract steps and answer from LLM response.

    Returns (steps, answer). Falls back to entire response as one step if
    the model doesn't follow the format.
    """
    steps = [m.strip() for m in _STEP_RE.findall(text)]
    answer_match = _ANSWER_RE.search(text)
    answer = answer_match.group(1).strip() if answer_match else text.strip()
    if not steps:
        steps = [text.strip()]
    return steps, answer


class LLMWorker(BaseWorker):
    def __init__(
        self,
        worker_id: str,
        bus: object,
        task_types: list[str] | None = None,
        *,
        model: str = "qwen2.5:1.5b",
        client: InferenceClient,
        skill_retriever: object | None = None,
    ) -> None:
        super().__init__(worker_id, bus, task_types or ["coding"])
        self.model = model
        self._client = client
        self._active_version: str | None = None
        self._skill_retriever = skill_retriever

    async def process(self, task: TaskEvent) -> ResultEvent:
        logger.info("LLMWorker %s calling model %s", self.worker_id, self.model)

        system_prompt = SYSTEM_PROMPT
        if self._skill_retriever is not None:
            try:
                skills = self._skill_retriever.retrieve(task.prompt)
                skills_section = self._skill_retriever.format_skills_prompt(skills)
                if skills_section:
                    system_prompt = f"{skills_section}\n{SYSTEM_PROMPT}"
            except Exception:
                logger.warning("Skill retrieval failed, using default prompt", exc_info=True)

        raw_text = await self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task.prompt},
            ],
        )
        steps, answer = _parse_steps(raw_text)
        return ResultEvent(
            task_id=task.task_id,
            worker_id=self.worker_id,
            result=answer,
            status=TaskStatus.SUCCESS,
            steps=steps,
        )

    async def reload_model(self, event: ModelUpdateEvent) -> None:
        old_model = self.model
        self.model = event.model_version
        self._active_version = event.model_version
        logger.info(
            "LLMWorker %s switched model %s -> %s",
            self.worker_id, old_model, self.model,
        )
