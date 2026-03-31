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

    if cfg.trainer_backend == "tinker":
        from src.training.tinker_backend import TinkerBackend

        trainer = TinkerBackend(
            api_key=cfg.tinker_api_key,
            base_url=cfg.tinker_base_url,
            model_name=cfg.training_model,
            learning_rate=cfg.training_lr,
            clip_range=cfg.training_clip_epsilon,
        )
        logger.info("Using Tinker backend with model=%s", cfg.training_model)
        return trainer

    if cfg.trainer_backend == "dapo":
        try:
            from src.training.trainer import DAPOTrainer

            trainer = DAPOTrainer(
                model_name=cfg.training_model,
                lr=cfg.training_lr,
                clip_epsilon=cfg.training_clip_epsilon,
                clip_epsilon_high=cfg.training_clip_epsilon_high,
                kl_beta=cfg.training_kl_beta,
                beta_entropy=cfg.dapo_entropy_beta,
                min_reward_threshold=cfg.dapo_min_reward_threshold,
                checkpoint_dir=cfg.training_checkpoint_dir,
                lora_rank=cfg.training_lora_rank,
                device=cfg.training_device,
            )
            logger.info("Using DAPOTrainer with model=%s", cfg.training_model)
            return trainer
        except (ImportError, Exception) as exc:
            logger.info("DAPOTrainer not available (%s), falling back", exc)

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

        base_scorer = LLMJudgeScorer(
            model=cfg.llm_model, client=inference_client, num_votes=cfg.prm_num_votes,
        )

        # Build CombinedScorer if HaluGate or Trained PRM is enabled
        scorer: object = base_scorer
        if cfg.halugate_enabled or cfg.trained_prm_enabled:
            from src.rewards.combined_scorer import CombinedScorer

            scorers: dict = {"prm": base_scorer}
            if cfg.halugate_enabled:
                from src.rewards.halugate_scorer import HaluGateScorer

                scorers["halugate"] = HaluGateScorer(
                    client=inference_client, model=cfg.halugate_model,
                )
                logger.info("HaluGate scorer enabled (model=%s)", cfg.halugate_model)
            if cfg.trained_prm_enabled and cfg.trained_prm_checkpoint:
                from src.rewards.trained_prm import TrainedPRMScorer

                scorers["prm"] = TrainedPRMScorer(
                    model_name=cfg.trained_prm_model,
                    device=cfg.training_device,
                    checkpoint_path=cfg.trained_prm_checkpoint,
                )
                logger.info("Trained PRM scorer enabled (checkpoint=%s)", cfg.trained_prm_checkpoint)
            scorer = CombinedScorer(scorers=scorers)

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

        # Trajectory store (optional persistence)
        trajectory_store = None
        if cfg.trajectory_store_enabled:
            from src.training.trajectory_store import TrajectoryStore

            trajectory_store = TrajectoryStore(db_path=cfg.trajectory_store_path)
            logger.info("TrajectoryStore enabled (path=%s)", cfg.trajectory_store_path)

        training_loop = TrainingLoop(
            bus, trainer, buffer, trajectory_store=trajectory_store,
        )
        await training_loop.start()
        logger.info("TrainingLoop started (backend=%s)", cfg.trainer_backend)

    logger.info("Training service running — waiting for events")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
