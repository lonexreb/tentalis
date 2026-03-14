"""Tests for the intercept proxy — mocked HTTP backend + EventBus."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.events.topics import SESSIONS
from src.events.types import SessionEvent
from src.intercept.proxy import create_proxy_app


@pytest.fixture
def mock_bus():
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def app(mock_bus):
    return create_proxy_app(mock_bus, backend_url="http://fake-backend:11434")


class TestInterceptProxy:
    async def test_health_endpoint(self, app):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["proxy"] is True

    async def test_chat_completions_forwards_and_logs(self, app, mock_bus):
        from httpx import ASGITransport, AsyncClient

        # Mock the backend response
        backend_response = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "model": "test-model",
        }

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = backend_response
        mock_http_response.status_code = 200

        app.state.http_client = AsyncMock()
        app.state.http_client.post = AsyncMock(return_value=mock_http_response)

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "user": "worker-01",
        }

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/v1/chat/completions", json=request_body)

        assert resp.status_code == 200
        result = resp.json()
        assert result["choices"][0]["message"]["content"] == "Hello!"

        # Verify backend was called
        app.state.http_client.post.assert_called_once()

        # Verify SessionEvent was published
        mock_bus.publish.assert_called_once()
        call_args = mock_bus.publish.call_args
        assert call_args[0][0] == SESSIONS
        session_event = call_args[0][1]
        assert isinstance(session_event, SessionEvent)
        assert session_event.worker_id == "worker-01"
        assert session_event.model == "test-model"
        assert session_event.response == "Hello!"

    async def test_chat_completions_publishes_even_on_bus_failure(self, app, mock_bus):
        """If NATS publish fails, the proxy still returns the response."""
        from httpx import ASGITransport, AsyncClient

        backend_response = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = backend_response
        mock_http_response.status_code = 200

        app.state.http_client = AsyncMock()
        app.state.http_client.post = AsyncMock(return_value=mock_http_response)
        mock_bus.publish.side_effect = RuntimeError("NATS down")

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": []},
            )

        assert resp.status_code == 200

    async def test_logprobs_extraction(self, app, mock_bus):
        from httpx import ASGITransport, AsyncClient

        backend_response = {
            "choices": [
                {
                    "message": {"content": "test"},
                    "logprobs": {
                        "content": [
                            {"logprob": -0.5},
                            {"logprob": -1.2},
                        ]
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = backend_response
        mock_http_response.status_code = 200

        app.state.http_client = AsyncMock()
        app.state.http_client.post = AsyncMock(return_value=mock_http_response)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": []},
            )

        session_event = mock_bus.publish.call_args[0][1]
        assert session_event.token_logprobs == [-0.5, -1.2]
