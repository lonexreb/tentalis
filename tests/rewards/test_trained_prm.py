"""Tests for Trained PRM — reward head, scorer, and trainer."""

from __future__ import annotations

import pytest

from src.rewards.trained_prm import TrainedPRMScorer


def _torch_available() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


requires_torch = pytest.mark.skipif(
    not _torch_available(), reason="torch not installed",
)


def test_trained_prm_scorer_has_score_steps():
    """Verify TrainedPRMScorer has the expected method."""
    assert hasattr(TrainedPRMScorer, "score_steps")


def test_prm_training_config_defaults():
    from src.rewards.prm_trainer import PRMTrainingConfig

    config = PRMTrainingConfig()
    assert config.lr == 1e-4
    assert config.epochs == 3
    assert config.batch_size == 32
    assert config.validation_split == 0.1


def test_prm_training_config_custom():
    from src.rewards.prm_trainer import PRMTrainingConfig

    config = PRMTrainingConfig(lr=5e-5, epochs=10, device="cuda")
    assert config.lr == 5e-5
    assert config.epochs == 10
    assert config.device == "cuda"


@requires_torch
def test_reward_head_output_shape():
    import torch
    from src.rewards.trained_prm import RewardHead

    head = RewardHead(hidden_dim=128)
    x = torch.randn(4, 128)
    out = head(x)
    assert out.shape == (4, 1)
    assert (out >= 0).all() and (out <= 1).all()  # sigmoid output


@requires_torch
def test_reward_head_gradient_flows():
    import torch
    from src.rewards.trained_prm import RewardHead

    head = RewardHead(hidden_dim=64)
    x = torch.randn(2, 64, requires_grad=True)
    out = head(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


@requires_torch
def test_reward_head_save_load_roundtrip(tmp_path):
    import torch
    from src.rewards.trained_prm import RewardHead

    head1 = RewardHead(hidden_dim=32)
    x = torch.randn(2, 32)
    out1 = head1(x)

    # Save
    torch.save(head1.state_dict(), tmp_path / "head.pt")

    # Load into new head
    head2 = RewardHead(hidden_dim=32)
    head2.load_state_dict(torch.load(tmp_path / "head.pt", weights_only=True))
    out2 = head2(x)

    assert torch.allclose(out1, out2)


@requires_torch
def test_prm_trainer_prepare_dataset():
    from dataclasses import dataclass

    from src.rewards.prm_trainer import PRMTrainer, PRMTrainingConfig

    @dataclass
    class FakeTrajectory:
        prompt: str = "Solve 2+2"
        steps: list = None
        step_scores: list = None

        def __post_init__(self):
            if self.steps is None:
                self.steps = ["Add 2+2", "Get 4"]
            if self.step_scores is None:
                self.step_scores = [0.7, 0.9]

    class FakeStore:
        def query(self, limit=1000):
            return [FakeTrajectory() for _ in range(5)]

        def count(self):
            return 5

    config = PRMTrainingConfig(validation_split=0.2)
    trainer = PRMTrainer(config=config, trajectory_store=FakeStore())
    train, val = trainer.prepare_dataset()
    # 5 trajectories * 2 steps = 10 examples
    assert len(train) + len(val) == 10
    assert len(val) >= 1  # at least 1 val example
    assert all("prompt" in ex and "step" in ex and "label" in ex for ex in train)
