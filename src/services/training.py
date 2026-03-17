"""Standalone training service: PRM Evaluator + Training Loop.

Run with: python -m src.services.training
"""

from __future__ import annotations

import asyncio
import logging

from src.config import Config
from src.events.bus import EventBus
from src.inference.client import create_client

logger = logging.getLogger(__name__)


def _create_trainer(cfg: Config) -> object:
    """Select and instantiate the training backend based on config."""
    if cfg.trainer_backend == "openrlhf":
        from src.training.openrlhf_backend import OpenRLHFBackend

        trainer = OpenRLHFBackend(
            model_name=cfg.training_model,
            output_dir=cfg.training_checkpoint_dir,
            learning_rate=cfg.training_lr,
            kl_coeff=cfg.training_kl_beta,
            clip_range=cfg.training_clip_epsilon,
        )
        logger.info("Using OpenRLHF backend with model=%s", cfg.training_model)
        return trainer

    # Standalone backend: try GRPOTrainer, fall back to MockTrainer
    try:
        from src.training.trainer import GRPOTrainer

        trainer = GRPOTrainer(
            model_name=cfg.training_model,
            lr=cfg.training_lr,
            clip_epsilon=cfg.training_clip_epsilon,
            kl_beta=cfg.training_kl_beta,
            checkpoint_dir=cfg.training_checkpoint_dir,
            lora_rank=cfg.training_lora_rank,
            device=cfg.training_device,
        )
        logger.info("Using GRPOTrainer with model=%s", cfg.training_model)
        return trainer
    except (ImportError, Exception) as exc:
        logger.info("GRPOTrainer not available (%s), using MockTrainer", exc)
        from src.training.trainer import MockTrainer

        return MockTrainer()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
    )
    cfg = Config()

    bus = EventBus()
    await bus.connect(cfg.nats_url)

    inference_client = create_client(
        backend=cfg.inference_backend,
        base_url=cfg.inference_base_url or cfg.ollama_host,
        api_key=cfg.inference_api_key,
    )

    # PRM Evaluator
    try:
        from src.rewards.prm_evaluator import PRMEvaluator
        from src.rewards.scorer import LLMJudgeScorer

        scorer = LLMJudgeScorer(model=cfg.llm_model, client=inference_client)
        evaluator = PRMEvaluator(bus, scorer)
        await evaluator.start(["coding"])
        logger.info("PRMEvaluator started")
    except (ImportError, Exception) as exc:
        logger.warning("Could not start PRMEvaluator: %s", exc)

    # Training Loop — select backend
    trainer = _create_trainer(cfg)

    if trainer is not None:
        from src.training.bridge import RolloutBuffer
        from src.training.loop import TrainingLoop

        buffer = RolloutBuffer(
            batch_size=cfg.training_batch_size,
            group_size=cfg.training_group_size,
        )
        training_loop = TrainingLoop(bus, trainer, buffer)
        await training_loop.start()
        logger.info("TrainingLoop started (backend=%s)", cfg.trainer_backend)

    logger.info("Training service running — waiting for events")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
