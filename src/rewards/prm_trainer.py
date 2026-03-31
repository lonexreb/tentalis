"""PRMTrainer — trains a RewardHead on accumulated trajectory data."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PRMTrainingConfig(BaseModel):
    """Configuration for PRM training."""

    model_name: str = "Qwen/Qwen2.5-0.5B"
    lr: float = 1e-4
    epochs: int = 3
    batch_size: int = 32
    max_length: int = 512
    device: str = "cpu"
    checkpoint_dir: str = "prm_checkpoints"
    validation_split: float = 0.1


class PRMTrainer:
    """Trains a RewardHead on accumulated trajectory data from TrajectoryStore.

    Training pipeline:
    1. Query TrajectoryStore for trajectories.
    2. Build (prompt, step, score) triples from step_scores.
    3. Split into train/val.
    4. Fine-tune RewardHead with MSE loss.
    5. Save checkpoint.
    """

    def __init__(
        self,
        config: PRMTrainingConfig,
        trajectory_store: object,
    ) -> None:
        self._config = config
        self._store = trajectory_store
        self._prm = None

    def prepare_dataset(self) -> tuple[list[dict], list[dict]]:
        """Build training examples from trajectory store.

        Each example: {"prompt": str, "step": str, "label": float}
        """
        trajectories = self._store.query(limit=100000)

        examples: list[dict] = []
        for traj in trajectories:
            context = traj.prompt
            for i, (step, score) in enumerate(
                zip(traj.steps, traj.step_scores)
            ):
                examples.append({
                    "prompt": context,
                    "step": step,
                    "label": score,
                })
                context = f"{context}\n{step}"

        if not examples:
            return [], []

        # Split
        split_idx = max(1, int(len(examples) * (1 - self._config.validation_split)))
        return examples[:split_idx], examples[split_idx:]

    def train(self) -> dict[str, float]:
        """Run training loop. Returns metrics dict."""
        import torch
        from torch.utils.data import DataLoader

        from src.rewards.trained_prm import TrainedPRM

        train_data, val_data = self.prepare_dataset()
        if not train_data:
            return {"error": "no training data", "num_examples": 0}

        # Initialize PRM
        self._prm = TrainedPRM(
            model_name=self._config.model_name,
            device=self._config.device,
        )
        self._prm._ensure_loaded()
        self._prm._reward_head.train()

        optimizer = torch.optim.AdamW(
            self._prm._reward_head.parameters(), lr=self._config.lr,
        )
        loss_fn = torch.nn.MSELoss()

        total_train_loss = 0.0
        num_batches = 0

        for epoch in range(self._config.epochs):
            # Simple batching
            for i in range(0, len(train_data), self._config.batch_size):
                batch = train_data[i : i + self._config.batch_size]

                prompts = [ex["prompt"] for ex in batch]
                steps = [ex["step"] for ex in batch]
                labels = torch.tensor(
                    [ex["label"] for ex in batch],
                    device=self._config.device,
                )

                # Get predictions
                texts = [f"{p}\n{s}" for p, s in zip(prompts, steps)]
                enc = self._prm._tokenizer(
                    texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=self._config.max_length,
                ).to(self._config.device)

                with torch.no_grad():
                    outputs = self._prm._base_model(**enc)
                    hidden = outputs.hidden_states[-1]
                    lengths = enc.attention_mask.sum(dim=1) - 1
                    last_hidden = hidden[
                        torch.arange(hidden.size(0)), lengths
                    ]

                preds = self._prm._reward_head(last_hidden).squeeze(-1)
                loss = loss_fn(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                num_batches += 1

            logger.info("Epoch %d/%d complete", epoch + 1, self._config.epochs)

        # Validation
        val_loss = 0.0
        if val_data:
            self._prm._reward_head.eval()
            val_preds: list[float] = []
            val_labels: list[float] = []

            for i in range(0, len(val_data), self._config.batch_size):
                batch = val_data[i : i + self._config.batch_size]
                prompts = [ex["prompt"] for ex in batch]
                steps = [ex["step"] for ex in batch]

                preds = self._prm.predict_batch(prompts, steps)
                val_preds.extend(preds)
                val_labels.extend(ex["label"] for ex in batch)

            if val_preds:
                val_loss = sum(
                    (p - l) ** 2 for p, l in zip(val_preds, val_labels)
                ) / len(val_preds)

        return {
            "train_loss": total_train_loss / max(num_batches, 1),
            "val_loss": val_loss,
            "num_examples": len(train_data),
            "num_val": len(val_data),
            "epochs": self._config.epochs,
        }

    def save_checkpoint(self) -> str:
        """Save reward head weights."""
        path = str(Path(self._config.checkpoint_dir) / "prm_latest")
        if self._prm is not None:
            self._prm.save(path)
        return path
