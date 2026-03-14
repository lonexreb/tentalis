"""Inference intercept proxy — transparent FastAPI proxy that logs SessionEvents to NATS."""

from __future__ import annotations

import logging
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.events.bus import EventBus
from src.events.topics import SESSIONS
from src.events.types import SessionEvent

logger = logging.getLogger(__name__)


def create_proxy_app(
    bus: EventBus,
    backend_url: str = "http://localhost:11434",
) -> FastAPI:
    """Create a FastAPI app that proxies /v1/chat/completions and logs sessions."""
    app = FastAPI(title="Intercept Proxy")
    app.state.bus = bus
    app.state.backend_url = backend_url.rstrip("/")
    app.state.http_client = httpx.AsyncClient(timeout=120.0)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await app.state.http_client.aclose()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()
        client: httpx.AsyncClient = app.state.http_client
        base = app.state.backend_url

        # Forward to real backend
        response = await client.post(
            f"{base}/v1/chat/completions",
            json=body,
            headers={"Content-Type": "application/json"},
        )
        result = response.json()

        # Extract response content for logging
        response_text = ""
        if "choices" in result and result["choices"]:
            choice = result["choices"][0]
            msg = choice.get("message", {})
            response_text = msg.get("content", "")

        # Extract logprobs if available
        token_logprobs: list[float] = []
        if "choices" in result and result["choices"]:
            logprobs_data = result["choices"][0].get("logprobs")
            if logprobs_data and "content" in logprobs_data:
                token_logprobs = [
                    t.get("logprob", 0.0)
                    for t in logprobs_data["content"]
                    if isinstance(t, dict)
                ]

        # Build and publish SessionEvent
        worker_id = body.get("user", "")
        session = SessionEvent(
            session_id=str(uuid.uuid4()),
            worker_id=worker_id,
            model=body.get("model", ""),
            messages=body.get("messages", []),
            response=response_text,
            token_logprobs=token_logprobs,
        )
        try:
            await app.state.bus.publish(SESSIONS, session)
        except Exception:
            logger.warning("Failed to publish SessionEvent", exc_info=True)

        return JSONResponse(content=result, status_code=response.status_code)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "proxy": True}

    return app
