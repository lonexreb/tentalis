"""Inference intercept proxy — transparent FastAPI proxy that logs SessionEvents to NATS.

Supports session tracking via X-Session-Id headers and optional skill injection.
"""

from __future__ import annotations

import logging
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.events.bus import EventBus
from src.events.topics import SESSIONS, SESSION_END, SESSION_START
from src.events.types import SessionEndEvent, SessionEvent, SessionStartEvent
from src.intercept.session_manager import SessionManager

logger = logging.getLogger(__name__)


def create_proxy_app(
    bus: EventBus,
    backend_url: str = "http://localhost:11434",
    *,
    skill_retriever: object | None = None,
    session_timeout: float = 1800.0,
) -> FastAPI:
    """Create a FastAPI app that proxies /v1/chat/completions and logs sessions."""
    app = FastAPI(title="Intercept Proxy")
    app.state.bus = bus
    app.state.backend_url = backend_url.rstrip("/")
    app.state.http_client = httpx.AsyncClient(timeout=120.0)
    app.state.session_manager = SessionManager(timeout_seconds=session_timeout)
    app.state.skill_retriever = skill_retriever

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await app.state.http_client.aclose()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()
        client: httpx.AsyncClient = app.state.http_client
        base = app.state.backend_url
        sm: SessionManager = app.state.session_manager

        # Session tracking via header
        session_id = request.headers.get("x-session-id")
        worker_id = body.get("user", "")
        model = body.get("model", "")

        session, is_new = sm.get_or_create(
            session_id, worker_id=worker_id, model=model,
        )

        if is_new:
            try:
                await app.state.bus.publish(
                    SESSION_START,
                    SessionStartEvent(
                        session_id=session.session_id,
                        worker_id=worker_id,
                        model=model,
                    ),
                )
            except Exception:
                logger.warning("Failed to publish SessionStartEvent", exc_info=True)

        # Inject skills into the request if retriever is available
        if app.state.skill_retriever is not None:
            body = _inject_skills(body, app.state.skill_retriever)

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

        # Track turn in session
        sm.add_turn(session.session_id, body.get("messages", []), response_text)

        # Build and publish SessionEvent
        session_event = SessionEvent(
            session_id=session.session_id,
            worker_id=worker_id,
            model=model,
            messages=body.get("messages", []),
            response=response_text,
            token_logprobs=token_logprobs,
        )
        try:
            await app.state.bus.publish(SESSIONS, session_event)
        except Exception:
            logger.warning("Failed to publish SessionEvent", exc_info=True)

        # Return with session ID in response headers
        return JSONResponse(
            content=result,
            status_code=response.status_code,
            headers={"X-Session-Id": session.session_id},
        )

    @app.post("/v1/sessions/{session_id}/end")
    async def end_session(session_id: str) -> dict:
        sm: SessionManager = app.state.session_manager
        session = sm.end(session_id)
        if session is None:
            return {"status": "not_found", "session_id": session_id}

        try:
            await app.state.bus.publish(
                SESSION_END,
                SessionEndEvent(
                    session_id=session_id,
                    worker_id=session.worker_id,
                    turn_count=session.turn_count,
                ),
            )
        except Exception:
            logger.warning("Failed to publish SessionEndEvent", exc_info=True)

        return {
            "status": "ended",
            "session_id": session_id,
            "turn_count": session.turn_count,
        }

    @app.get("/health")
    async def health() -> dict:
        sm: SessionManager = app.state.session_manager
        return {
            "status": "ok",
            "proxy": True,
            "active_sessions": sm.active_count,
        }

    return app


def _inject_skills(body: dict, retriever: object) -> dict:
    """Inject retrieved skills into the system message of a chat request."""
    messages = body.get("messages", [])
    if not messages:
        return body

    # Find the user message to use as retrieval query
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return body

    query = user_msgs[-1].get("content", "")
    if not query:
        return body

    try:
        skills = retriever.retrieve(query)
        skills_text = retriever.format_skills_prompt(skills)
        if not skills_text:
            return body
    except Exception:
        logger.warning("Skill injection failed", exc_info=True)
        return body

    # Prepend skills to the system message
    new_messages = list(messages)
    if new_messages and new_messages[0].get("role") == "system":
        new_messages[0] = {
            **new_messages[0],
            "content": f"{skills_text}\n{new_messages[0]['content']}",
        }
    else:
        new_messages.insert(0, {"role": "system", "content": skills_text})

    return {**body, "messages": new_messages}
