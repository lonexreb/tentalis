"""OpenRLHF training backend — implements Trainer protocol via Ray-based OpenRLHF.

Bridges our event-driven rollout pipeline to OpenRLHF's production-grade GRPO
training (Ray + vLLM + DeepSpeed). Our custom GRPO math is replaced by
OpenRLHF's GPU-optimized implementation; we keep the Trainer protocol so
TrainingLoop doesn't know the difference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.events.types import TrainingRolloutEvent
from src.training.trainer import TrainStepResult

logger = logging.getLogger(__name__)


class OpenRLHFBackend:
    """Trainer protocol implementation backed by OpenRLHF.

    Collects rollouts from our NATS pipeline, exports them to JSONL,
    launches OpenRLHF training via Ray, and returns results as TrainStepResult.

    For CPU testing / environments without Ray+vLLM, falls back to a
    lightweight simulation that validates the data flow without actual
    GPU training.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        output_dir: str = "openrlhf_checkpoints",
        dataset_dir: str = "openrlhf_datasets",
        ray_num_gpus: int = 0,
        vllm_tensor_parallel: int = 1,
        deepspeed_stage: int = 2,
        max_epochs: int = 1,
        micro_batch_size: int = 4,
        learning_rate: float = 1e-5,
        kl_coeff: float = 0.1,
        clip_range: float = 0.2,
        use_dapo: bool = False,
        clip_range_high: float = 0.28,
        entropy_bonus: float = 0.0,
    ) -> None:
        self._model_name = model_name
        self._output_dir = Path(output_dir)
        self._dataset_dir = Path(dataset_dir)
        self._ray_num_gpus = ray_num_gpus
        self._vllm_tp = vllm_tensor_parallel
        self._ds_stage = deepspeed_stage
        self._max_epochs = max_epochs
        self._micro_batch_size = micro_batch_size
        self._lr = learning_rate
        self._kl_coeff = kl_coeff
        self._clip_range = clip_range
        self._use_dapo = use_dapo
        self._clip_range_high = clip_range_high
        self._entropy_bonus = entropy_bonus
        self._step_count = 0
        self._last_checkpoint: str | None = None
        self._ray_available: bool | None = None

    def _check_ray(self) -> bool:
        if self._ray_available is None:
            try:
                import ray  # noqa: F401
                self._ray_available = True
            except ImportError:
                self._ray_available = False
                logger.warning(
                    "Ray not available — OpenRLHF backend will run in "
                    "simulation mode (data flow validation only)"
                )
        return self._ray_available

    def _export_rollouts(
        self, batch: list[TrainingRolloutEvent], path: Path
    ) -> int:
        """Export rollouts to JSONL format consumable by OpenRLHF."""
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(path, "w") as f:
            for r in batch:
                record = {
                    "prompt": r.prompt,
                    "response": r.response,
                    "reward": r.outcome_score,
                    "step_scores": r.step_scores,
                    "worker_id": r.worker_id,
                    "task_id": r.task_id,
                }
                f.write(json.dumps(record) + "\n")
                count += 1
        return count

    async def train_step(
        self, batch: list[TrainingRolloutEvent]
    ) -> TrainStepResult:
        """Run one training step via OpenRLHF or simulation."""
        self._step_count += 1

        # Export rollouts to JSONL
        dataset_path = self._dataset_dir / f"batch_{self._step_count:06d}.jsonl"
        num_exported = self._export_rollouts(batch, dataset_path)
        logger.info(
            "Exported %d rollouts to %s", num_exported, dataset_path
        )

        # Compute batch stats for reporting
        scores = [r.outcome_score for r in batch]
        mean_score = sum(scores) / len(scores) if scores else 0.0

        if self._check_ray():
            return await self._train_with_ray(batch, dataset_path, mean_score)
        else:
            return self._simulate_training(batch, mean_score)

    async def _train_with_ray(
        self,
        batch: list[TrainingRolloutEvent],
        dataset_path: Path,
        mean_score: float,
    ) -> TrainStepResult:
        """Launch actual OpenRLHF training via Ray subprocess."""
        import asyncio

        checkpoint_dir = self._output_dir / f"step_{self._step_count:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "-m", "openrlhf.cli.train_grpo_ray",
            "--pretrain", self._model_name,
            "--dataset", str(dataset_path),
            "--save_path", str(checkpoint_dir),
            "--max_epochs", str(self._max_epochs),
            "--micro_train_batch_size", str(self._micro_batch_size),
            "--learning_rate", str(self._lr),
            "--kl_coeff", str(self._kl_coeff),
            "--cliprange", str(self._clip_range),
        ]

        if self._ray_num_gpus > 0:
            cmd.extend(["--num_gpus", str(self._ray_num_gpus)])
        if self._vllm_tp > 1:
            cmd.extend(["--vllm_tensor_parallel_size", str(self._vllm_tp)])
        if self._ds_stage > 0:
            cmd.extend(["--deepspeed_stage", str(self._ds_stage)])

        # DAPO support
        if self._use_dapo:
            cmd.extend(["--use_dapo", "--clip_range_high", str(self._clip_range_high)])
            if self._entropy_bonus > 0:
                cmd.extend(["--entropy_bonus", str(self._entropy_bonus)])

        logger.info("Launching OpenRLHF: %s", " ".join(cmd))

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            self._last_checkpoint = str(checkpoint_dir)
            logger.info("OpenRLHF training completed: %s", checkpoint_dir)

            # Parse training metrics from stdout if available
            loss = self._parse_loss_from_output(stdout.decode())
        else:
            logger.error(
                "OpenRLHF training failed (exit %d): %s",
                process.returncode,
                stderr.decode()[-500:] if stderr else "no stderr",
            )
            loss = -1.0
            self._last_checkpoint = None

        return TrainStepResult(
            loss=loss,
            mean_advantage=mean_score,
            std_advantage=0.0,
            checkpoint_path=self._last_checkpoint,
            step_count=self._step_count,
        )

    def _simulate_training(
        self,
        batch: list[TrainingRolloutEvent],
        mean_score: float,
    ) -> TrainStepResult:
        """Simulate training for CPU testing — validates data flow."""
        logger.info(
            "OpenRLHF simulation step %d: batch_size=%d, mean_score=%.3f",
            self._step_count, len(batch), mean_score,
        )

        # Create a checkpoint directory to validate the path works
        checkpoint_dir = self._output_dir / f"sim_step_{self._step_count:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._last_checkpoint = str(checkpoint_dir)

        # Write a metadata file so we can verify the checkpoint was created
        meta_path = checkpoint_dir / "training_metadata.json"
        meta_path.write_text(json.dumps({
            "step": self._step_count,
            "batch_size": len(batch),
            "mean_score": mean_score,
            "model_name": self._model_name,
            "backend": "openrlhf_simulation",
        }))

        return TrainStepResult(
            loss=1.0 / self._step_count,
            mean_advantage=mean_score,
            std_advantage=0.0,
            checkpoint_path=self._last_checkpoint,
            step_count=self._step_count,
        )

    @staticmethod
    def _parse_loss_from_output(output: str) -> float:
        """Best-effort parse training loss from OpenRLHF stdout."""
        for line in reversed(output.split("\n")):
            if "loss" in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if "loss" in part.lower() and i + 1 < len(parts):
                        try:
                            return float(parts[i + 1].strip(",: "))
                        except ValueError:
                            continue
        return 0.0

    def checkpoint_path(self) -> str | None:
        return self._last_checkpoint
