"""GUIWorker — screenshot + action pairs for GUI automation tasks."""

from __future__ import annotations

import base64
import logging

from src.events.bus import EventBus
from src.events.types import ResultEvent, TaskEvent, TaskStatus
from src.inference.client import InferenceClient
from src.workers.base import BaseWorker

logger = logging.getLogger(__name__)

ACTION_PROMPT = """\
You are a GUI automation agent. Given the task below, describe the next action \
to take as a JSON object with keys: "action" (click/type/scroll/wait), \
"target" (element description), "value" (text to type, if applicable).

Task: {task}
Previous actions: {previous_actions}

Output ONLY the JSON action object."""

MAX_STEPS = 10


class GUIWorker(BaseWorker):
    """Performs GUI tasks via screenshot → action loops.

    Steps are (screenshot_description, action) pairs. Uses an LLM to decide
    the next action based on the current screenshot state.

    In production, integrates with pyautogui for actual GUI control.
    This implementation uses the LLM to simulate the action planning loop.
    """

    environment_type = "gui"

    def __init__(
        self,
        worker_id: str,
        bus: EventBus,
        client: InferenceClient,
        model: str = "qwen2.5:1.5b",
        task_types: list[str] | None = None,
        *,
        max_steps: int = MAX_STEPS,
        screenshot_fn: object | None = None,
    ) -> None:
        super().__init__(worker_id, bus, task_types or ["gui"])
        self._client = client
        self._model = model
        self._max_steps = max_steps
        self._screenshot_fn = screenshot_fn  # Callable[[], bytes] for production

    async def process(self, task: TaskEvent) -> ResultEvent:
        steps: list[str] = []
        previous_actions: list[str] = []

        try:
            for i in range(self._max_steps):
                # Get next action from LLM
                action_text = await self._client.chat(
                    model=self._model,
                    messages=[
                        {
                            "role": "user",
                            "content": ACTION_PROMPT.format(
                                task=task.prompt,
                                previous_actions=previous_actions or "None",
                            ),
                        }
                    ],
                    json_mode=True,
                )

                step_desc = f"step_{i + 1}: {action_text}"
                steps.append(step_desc)
                previous_actions.append(action_text)

                # Check if the LLM signals completion
                if '"action"' in action_text and '"done"' in action_text.lower():
                    break

            return ResultEvent(
                task_id=task.task_id,
                worker_id=self.worker_id,
                result="\n".join(steps),
                status=TaskStatus.SUCCESS,
                steps=steps,
            )
        except Exception as exc:
            logger.exception("GUIWorker failed on task %s", task.task_id)
            return ResultEvent(
                task_id=task.task_id,
                worker_id=self.worker_id,
                result=str(exc),
                status=TaskStatus.FAILED,
                steps=steps,
            )

    @staticmethod
    def _encode_screenshot(data: bytes) -> str:
        return base64.b64encode(data).decode()
