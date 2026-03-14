"""Tests for PerWorkerAdapterRegistry."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.events.types import ModelUpdateEvent
from src.inference.adapter_registry import PerWorkerAdapterRegistry


@pytest.fixture
def mock_bus():
    bus = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture
def mock_lora():
    lora = AsyncMock()
    lora.load_adapter = AsyncMock()
    lora.unload_adapter = AsyncMock()
    return lora


@pytest.fixture
def registry(mock_bus, mock_lora):
    return PerWorkerAdapterRegistry(mock_bus, mock_lora)


class TestPerWorkerAdapterRegistry:
    async def test_start_subscribes(self, registry, mock_bus):
        await registry.start()
        mock_bus.subscribe.assert_called_once()

    async def test_targeted_update_registers_worker(self, registry, mock_lora):
        event = ModelUpdateEvent(
            model_version="v0001",
            checkpoint_path="/checkpoints/v0001",
            target_worker_id="w1",
        )
        await registry._handle_update(event)

        mock_lora.load_adapter.assert_called_once()
        assert registry.get_adapter_name("w1") == "worker-w1-v0001"
        assert registry.get_adapter_path("w1") == "/checkpoints/v0001"

    async def test_broadcast_update_loads_global(self, registry, mock_lora):
        event = ModelUpdateEvent(
            model_version="v0002",
            checkpoint_path="/checkpoints/v0002",
            target_worker_id=None,
        )
        await registry._handle_update(event)

        mock_lora.load_adapter.assert_called_once_with(
            "global-v0002", "/checkpoints/v0002"
        )

    async def test_update_unloads_previous(self, registry, mock_lora):
        # First update
        event1 = ModelUpdateEvent(
            model_version="v0001",
            checkpoint_path="/checkpoints/v0001",
            target_worker_id="w1",
        )
        await registry._handle_update(event1)

        # Second update
        event2 = ModelUpdateEvent(
            model_version="v0002",
            checkpoint_path="/checkpoints/v0002",
            target_worker_id="w1",
        )
        await registry._handle_update(event2)

        mock_lora.unload_adapter.assert_called_once_with("worker-w1-v0001")
        assert registry.get_adapter_name("w1") == "worker-w1-v0002"

    async def test_registered_workers(self, registry):
        for wid in ["w1", "w2"]:
            event = ModelUpdateEvent(
                model_version="v0001",
                checkpoint_path=f"/ckpt/{wid}",
                target_worker_id=wid,
            )
            await registry._handle_update(event)

        assert sorted(registry.registered_workers) == ["w1", "w2"]

    async def test_unknown_worker_returns_none(self, registry):
        assert registry.get_adapter_name("unknown") is None
        assert registry.get_adapter_path("unknown") is None
