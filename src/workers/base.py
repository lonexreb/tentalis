from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from src.events.bus import EventBus
from src.events.topics import MODEL_UPDATES, result_topic, task_topic
from src.events.types import ModelUpdateEvent, ResultEvent, TaskEvent, TaskStatus

logger = logging.getLogger(__name__)


class BaseWorker(ABC):
    environment_type: str = "chat"

    def __init__(self, worker_id: str, bus: EventBus, task_types: list[str]) -> None:
        self.worker_id = worker_id
        self.bus = bus
        self.task_types = task_types

    async def start(self) -> None:
        for tt in self.task_types:
            await self.bus.subscribe(task_topic(tt), TaskEvent, self._handle_task)
        await self.bus.subscribe(MODEL_UPDATES, ModelUpdateEvent, self._handle_model_update)
        logger.info("Worker %s started, listening for %s", self.worker_id, self.task_types)

    async def _handle_task(self, task: TaskEvent) -> None:
        logger.info("Worker %s received task %s", self.worker_id, task.task_id)
        t0 = time.monotonic()
        try:
            result = await self.process(task)
            result.prompt = task.prompt
            result.elapsed_seconds = time.monotonic() - t0
            result.model_version = getattr(self, "_active_version", None)
        except Exception as exc:
            logger.exception("Worker %s failed task %s", self.worker_id, task.task_id)
            result = ResultEvent(
                task_id=task.task_id,
                worker_id=self.worker_id,
                prompt=task.prompt,
                result=str(exc),
                status=TaskStatus.FAILED,
                elapsed_seconds=time.monotonic() - t0,
            )
        await self.bus.publish(result_topic(task.task_type), result)

    async def _handle_model_update(self, event: ModelUpdateEvent) -> None:
        # Filter by target_worker_id if set
        if event.target_worker_id and event.target_worker_id != self.worker_id:
            logger.debug(
                "Worker %s ignoring model update targeted at %s",
                self.worker_id, event.target_worker_id,
            )
            return
        logger.info("Worker %s received model update: %s", self.worker_id, event.model_version)
        await self.reload_model(event)

    async def reload_model(self, event: ModelUpdateEvent) -> None:
        """Override in subclasses to react to model updates. Default no-op."""

    @abstractmethod
    async def process(self, task: TaskEvent) -> ResultEvent: ...
