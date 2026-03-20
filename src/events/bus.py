from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import nats
from nats.aio.client import Client
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class EventBus:
    def __init__(self) -> None:
        self._nc: Client | None = None
        self._subscriptions: list = []

    async def connect(self, url: str = "nats://localhost:4222") -> None:
        self._nc = await nats.connect(url)
        logger.info("Connected to NATS at %s", url)

    async def close(self) -> None:
        if self._nc and self._nc.is_connected:
            await self._nc.drain()
            logger.info("NATS connection drained and closed")

    async def publish(self, topic: str, event: BaseModel) -> None:
        if not self._nc:
            raise RuntimeError("EventBus not connected")
        data = event.model_dump_json().encode()
        await self._nc.publish(topic, data)
        logger.debug("Published to %s: %s", topic, event)

    async def subscribe(
        self,
        topic: str,
        event_type: type[T],
        handler: Callable[[T], Awaitable[None]],
    ) -> None:
        if not self._nc:
            raise RuntimeError("EventBus not connected")

        async def _msg_handler(msg: object) -> None:
            try:
                event = event_type.model_validate_json(msg.data)
                await handler(event)
            except Exception:
                logger.exception("Error handling message on %s", topic)

        sub = await self._nc.subscribe(topic, cb=_msg_handler)
        self._subscriptions.append(sub)
        logger.info("Subscribed to %s", topic)

    async def subscribe_raw(
        self,
        topic: str,
        handler: Callable[[str, bytes], Awaitable[None]],
    ) -> None:
        """Subscribe to raw bytes on a topic (used by audit logger)."""
        if not self._nc:
            raise RuntimeError("EventBus not connected")

        async def _msg_handler(msg: object) -> None:
            await handler(topic, msg.data)

        sub = await self._nc.subscribe(topic, cb=_msg_handler)
        self._subscriptions.append(sub)
        logger.info("Subscribed (raw) to %s", topic)
