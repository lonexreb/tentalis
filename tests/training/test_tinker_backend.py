"""Tests for TinkerBackend — Trainer protocol conformance with mocked Tinker SDK."""

from unittest.mock import MagicMock, patch

import pytest

from src.events.types import TrainingRolloutEvent
from src.training.tinker_backend import TinkerBackend
from src.training.trainer import Trainer, TrainStepResult


def _make_rollout(prompt: str = "solve x+1=2", score: float = 0.8) -> TrainingRolloutEvent:
    return TrainingRolloutEvent(
        task_id="t1",
        worker_id="w1",
        prompt=prompt,
        response="x=1",
        steps=["step 1"],
        step_scores=[score],
        outcome_score=score,
    )


class TestTinkerBackendProtocol:
    def test_has_train_step(self):
        """TinkerBackend has train_step method matching Trainer protocol."""
        backend = TinkerBackend(api_key="test")
        assert hasattr(backend, "train_step")
        assert hasattr(backend, "checkpoint_path")

    def test_checkpoint_path_returns_none(self):
        backend = TinkerBackend(api_key="test")
        assert backend.checkpoint_path() is None


class TestTinkerBackendTraining:
    @pytest.fixture
    def mock_tinker_client(self):
        client = MagicMock()
        client.forward_backward.return_value = {
            "loss": 0.5,
            "mean_advantage": 0.1,
            "std_advantage": 0.05,
        }
        client.optim_step.return_value = {
            "checkpoint_id": "ckpt-001",
        }
        return client

    @pytest.fixture
    def backend(self, mock_tinker_client):
        b = TinkerBackend(
            api_key="test-key",
            base_url="https://fake.tinker.ai",
            model_name="test-model",
        )
        b._client = mock_tinker_client
        return b

    async def test_train_step_returns_result(self, backend, mock_tinker_client):
        batch = [_make_rollout(), _make_rollout(score=0.6)]
        result = await backend.train_step(batch)

        assert isinstance(result, TrainStepResult)
        assert result.loss == 0.5
        assert result.mean_advantage == 0.1
        assert result.step_count == 1
        assert result.checkpoint_path == "ckpt-001"

    async def test_train_step_calls_tinker_api(self, backend, mock_tinker_client):
        batch = [_make_rollout()]
        await backend.train_step(batch)

        mock_tinker_client.forward_backward.assert_called_once()
        mock_tinker_client.optim_step.assert_called_once()

        fb_call = mock_tinker_client.forward_backward.call_args
        assert fb_call.kwargs["model"] == "test-model"
        assert len(fb_call.kwargs["samples"]) == 1

    async def test_step_count_increments(self, backend):
        batch = [_make_rollout()]
        r1 = await backend.train_step(batch)
        r2 = await backend.train_step(batch)
        assert r1.step_count == 1
        assert r2.step_count == 2

    async def test_handles_non_dict_responses(self, backend, mock_tinker_client):
        mock_tinker_client.forward_backward.return_value = None
        mock_tinker_client.optim_step.return_value = None

        result = await backend.train_step([_make_rollout()])
        assert result.loss == 0.0
        assert result.checkpoint_path is None

    def test_ensure_client_raises_without_tinker(self):
        backend = TinkerBackend(api_key="test")
        with pytest.raises(ImportError, match="tinker package not installed"):
            backend._ensure_client()
