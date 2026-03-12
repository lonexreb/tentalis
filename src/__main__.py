"""Demo entry point: python -m src"""

import asyncio
import logging

from src.config import Config
from src.events.bus import EventBus
from src.events.topics import TRAINING_ROLLOUTS
from src.events.types import TrainingRolloutEvent
from src.manager.manager import Manager

logger = logging.getLogger(__name__)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = Config()

    bus = EventBus()
    await bus.connect(cfg.nats_url)

    # Try LLMWorker if ollama is available, else fall back to EchoWorker
    try:
        from src.workers.llm_worker import LLMWorker

        import ollama  # noqa: F401

        worker = LLMWorker(
            cfg.worker_id,
            bus,
            model=cfg.llm_model,
            ollama_host=cfg.ollama_host,
        )
        print(f"Using LLMWorker with model={cfg.llm_model}")
    except (ImportError, Exception) as exc:
        logger.info("Ollama not available (%s), falling back to EchoWorker", exc)
        from src.workers.echo_worker import EchoWorker

        worker = EchoWorker(cfg.worker_id, bus)
        print("Using EchoWorker (fallback)")

    # Set up PRM evaluator
    try:
        from src.rewards.scorer import LLMJudgeScorer
        from src.rewards.prm_evaluator import PRMEvaluator

        scorer = LLMJudgeScorer(
            model=cfg.llm_model, ollama_host=cfg.ollama_host
        )
        evaluator = PRMEvaluator(bus, scorer)
        await evaluator.start(["coding"])
        print("PRMEvaluator started")
    except (ImportError, Exception) as exc:
        logger.info("Could not start PRMEvaluator: %s", exc)
        evaluator = None

    # Set up training loop
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
        print(f"Using GRPOTrainer with model={cfg.training_model}")
    except (ImportError, Exception) as exc:
        logger.info("torch/transformers/peft not available (%s), using MockTrainer", exc)
        from src.training.trainer import MockTrainer

        trainer = MockTrainer()
        print("Using MockTrainer (fallback)")

    from src.training.bridge import RolloutBuffer
    from src.training.loop import TrainingLoop

    buffer = RolloutBuffer(
        batch_size=cfg.training_batch_size,
        group_size=cfg.training_group_size,
    )
    training_loop = TrainingLoop(bus, trainer, buffer)
    await training_loop.start()
    print("TrainingLoop started")

    # Capture rollouts for display
    rollout_event = asyncio.Event()
    captured_rollout: list[TrainingRolloutEvent] = []

    async def _on_rollout(rollout: TrainingRolloutEvent) -> None:
        captured_rollout.append(rollout)
        rollout_event.set()

    await bus.subscribe(TRAINING_ROLLOUTS, TrainingRolloutEvent, _on_rollout)

    manager = Manager(cfg.manager_id, bus)
    await manager.start()
    await worker.start()

    print("\n--- Assigning task ---")
    task = await manager.assign_task("coding", "Write a fibonacci function in Python")
    print(f"Task ID: {task.task_id}")

    result = await manager.wait_for_result(task.task_id, timeout=cfg.task_timeout_seconds)
    print("\n--- Result ---")
    print(f"Status: {result.status.value}")
    print(f"Result: {result.result[:200]}")
    print(f"Steps:  {result.steps}")
    print(f"Time:   {result.elapsed_seconds:.4f}s")

    # Wait briefly for PRM scoring
    if evaluator:
        try:
            await asyncio.wait_for(rollout_event.wait(), timeout=30.0)
            rollout = captured_rollout[0]
            print("\n--- PRM Scores ---")
            for i, (step, score) in enumerate(
                zip(rollout.steps, rollout.step_scores), 1
            ):
                print(f"  Step {i}: {score:.3f} — {step[:80]}")
            print(f"  Outcome: {rollout.outcome_score:.3f}")
        except asyncio.TimeoutError:
            print("\n--- PRM scoring timed out ---")

    await manager.publish_feedback(result, score=0.95, text="Task completed")
    print("\n--- Feedback published (score=0.95) ---")

    await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
