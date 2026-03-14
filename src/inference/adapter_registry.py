"""PerWorkerAdapterRegistry — maps worker_id to LoRA adapter paths."""

from __future__ import annotations

import logging

from src.events.bus import EventBus
from src.events.topics import MODEL_UPDATES
from src.events.types import ModelUpdateEvent
from src.inference.vllm_lora import VLLMLoRAManager

logger = logging.getLogger(__name__)


class PerWorkerAdapterRegistry:
    """Manages per-worker LoRA adapters on a vLLM server.

    Listens for ModelUpdateEvents with target_worker_id set.
    Maintains a mapping of worker_id → (adapter_name, adapter_path).
    Integrates with VLLMLoRAManager for hot-swapping.
    """

    def __init__(
        self,
        bus: EventBus,
        lora_manager: VLLMLoRAManager,
    ) -> None:
        self._bus = bus
        self._lora = lora_manager
        self._registry: dict[str, tuple[str, str]] = {}  # worker_id → (name, path)

    async def start(self) -> None:
        await self._bus.subscribe(
            MODEL_UPDATES, ModelUpdateEvent, self._handle_update
        )
        logger.info("PerWorkerAdapterRegistry started")

    async def _handle_update(self, event: ModelUpdateEvent) -> None:
        worker_id = event.target_worker_id
        if not worker_id:
            # Broadcast update — load as default adapter
            adapter_name = f"global-{event.model_version}"
            await self._load_adapter(adapter_name, event.checkpoint_path)
            return

        adapter_name = f"worker-{worker_id}-{event.model_version}"

        # Unload previous adapter for this worker if exists
        old = self._registry.get(worker_id)
        if old:
            try:
                await self._lora.unload_adapter(old[0])
            except Exception:
                logger.warning("Failed to unload old adapter %s", old[0], exc_info=True)

        await self._load_adapter(adapter_name, event.checkpoint_path)
        self._registry[worker_id] = (adapter_name, event.checkpoint_path)
        logger.info(
            "Registered adapter %s for worker %s", adapter_name, worker_id
        )

    async def _load_adapter(self, name: str, path: str) -> None:
        try:
            await self._lora.load_adapter(name, path)
        except Exception:
            logger.error("Failed to load adapter %s from %s", name, path, exc_info=True)

    def get_adapter_name(self, worker_id: str) -> str | None:
        entry = self._registry.get(worker_id)
        return entry[0] if entry else None

    def get_adapter_path(self, worker_id: str) -> str | None:
        entry = self._registry.get(worker_id)
        return entry[1] if entry else None

    @property
    def registered_workers(self) -> list[str]:
        return list(self._registry.keys())
