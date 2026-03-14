"""Entrypoint: python -m src.intercept"""

from __future__ import annotations

import asyncio
import logging

import uvicorn

from src.config import Config
from src.events.bus import EventBus
from src.intercept.proxy import create_proxy_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    cfg = Config()
    bus = EventBus()
    await bus.connect(cfg.nats_url)

    app = create_proxy_app(bus, backend_url=cfg.intercept_backend_url)

    config = uvicorn.Config(app, host="0.0.0.0", port=cfg.intercept_port, log_level="info")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
