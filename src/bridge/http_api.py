"""HTTP API endpoints for OpenClaw agents to interact with NATS."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiohttp import web
from pydantic import ValidationError

from src.events.bus import EventBus
from src.events.topics import FEEDBACK_SCORED, SESSIONS, result_topic, task_topic
from src.events.types import (
    FeedbackEvent,
    ResultEvent,
    SessionEvent,
    TaskEvent,
    TaskStatus,
)

logger = logging.getLogger(__name__)


def create_app(
    bus: EventBus,
    results: dict[str, ResultEvent],
    waiters: dict[str, asyncio.Event],
) -> web.Application:
    app = web.Application()
    app["bus"] = bus
    app["results"] = results
    app["waiters"] = waiters

    app.router.add_post("/tasks/assign", handle_assign_task)
    app.router.add_post("/tasks/result", handle_submit_result)
    app.router.add_post("/feedback", handle_submit_feedback)
    app.router.add_get("/tasks/{task_id}/status", handle_task_status)
    app.router.add_post("/sessions/log", handle_session_log)
    app.router.add_get("/training/status", handle_training_status)
    app.router.add_get("/health", handle_health)

    return app


async def handle_assign_task(request: web.Request) -> web.Response:
    """POST /tasks/assign — Manager agent publishes a TaskEvent."""
    bus: EventBus = request.app["bus"]

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    try:
        event = TaskEvent(**body)
    except (ValidationError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=422)

    topic = task_topic(event.task_type)
    await bus.publish(topic, event)
    logger.info("Bridge published TaskEvent %s to %s", event.task_id, topic)

    # Register a waiter so status polling works
    request.app["waiters"][event.task_id] = asyncio.Event()

    return web.json_response(
        {"task_id": event.task_id, "topic": topic, "status": "published"},
        status=201,
    )


async def handle_submit_result(request: web.Request) -> web.Response:
    """POST /tasks/result — Worker agent publishes a ResultEvent."""
    bus: EventBus = request.app["bus"]

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    # Coerce status string to enum
    if "status" in body and isinstance(body["status"], str):
        try:
            body["status"] = TaskStatus(body["status"])
        except ValueError:
            pass

    try:
        event = ResultEvent(**body)
    except (ValidationError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=422)

    # Infer task_type from metadata or default to "coding"
    task_type = body.get("task_type", "coding")
    topic = result_topic(task_type)
    await bus.publish(topic, event)
    logger.info("Bridge published ResultEvent %s to %s", event.task_id, topic)

    return web.json_response(
        {"task_id": event.task_id, "topic": topic, "status": "published"},
        status=201,
    )


async def handle_submit_feedback(request: web.Request) -> web.Response:
    """POST /feedback — Manager agent publishes a FeedbackEvent."""
    bus: EventBus = request.app["bus"]

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    try:
        event = FeedbackEvent(**body)
    except (ValidationError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=422)

    await bus.publish(FEEDBACK_SCORED, event)
    logger.info("Bridge published FeedbackEvent for task %s", event.task_id)

    return web.json_response(
        {"task_id": event.task_id, "status": "published"},
        status=201,
    )


async def handle_task_status(request: web.Request) -> web.Response:
    """GET /tasks/{task_id}/status — Poll for result."""
    task_id = request.match_info["task_id"]
    results: dict[str, ResultEvent] = request.app["results"]

    # If result already available, return immediately
    result = results.get(task_id)
    if result:
        return web.json_response(
            {"task_id": task_id, "status": "completed", "result": result.model_dump(mode="json")},
        )

    # Wait up to 30s for result
    waiters: dict[str, asyncio.Event] = request.app["waiters"]
    waiter = waiters.get(task_id)
    if waiter:
        try:
            await asyncio.wait_for(waiter.wait(), timeout=30.0)
            result = results.get(task_id)
            if result:
                return web.json_response(
                    {"task_id": task_id, "status": "completed", "result": result.model_dump(mode="json")},
                )
        except asyncio.TimeoutError:
            pass

    return web.json_response(
        {"task_id": task_id, "status": "pending"},
    )


async def handle_session_log(request: web.Request) -> web.Response:
    """POST /sessions/log — Log a session event from the intercept proxy."""
    bus: EventBus = request.app["bus"]

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    try:
        event = SessionEvent(**body)
    except (ValidationError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=422)

    await bus.publish(SESSIONS, event)
    logger.info("Bridge published SessionEvent %s", event.session_id)

    return web.json_response(
        {"session_id": event.session_id, "status": "published"},
        status=201,
    )


async def handle_training_status(request: web.Request) -> web.Response:
    """GET /training/status — Report training pipeline status."""
    return web.json_response({
        "status": "ok",
        "components": {
            "prm_evaluator": "active",
            "training_loop": "active",
            "combined_rollout_builder": "active",
        },
    })


async def handle_health(request: web.Request) -> web.Response:
    """GET /health — Liveness check."""
    return web.json_response({"status": "ok"})
