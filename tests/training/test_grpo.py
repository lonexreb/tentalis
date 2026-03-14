"""Tests for GRPO math utilities."""

import pytest

from src.training.grpo import compute_group_advantages


class TestComputeGroupAdvantages:
    def test_empty_list(self):
        assert compute_group_advantages([]) == []

    def test_single_element(self):
        assert compute_group_advantages([0.5]) == [0.0]

    def test_identical_rewards(self):
        result = compute_group_advantages([0.7, 0.7, 0.7])
        assert result == [0.0, 0.0, 0.0]

    def test_two_elements(self):
        result = compute_group_advantages([1.0, 0.0])
        # mean=0.5, std=0.5 → [1.0, -1.0]
        assert pytest.approx(result[0], abs=1e-6) == 1.0
        assert pytest.approx(result[1], abs=1e-6) == -1.0

    def test_sum_is_zero(self):
        """Advantages should sum to ~0 (centered)."""
        result = compute_group_advantages([0.2, 0.5, 0.8, 0.3])
        assert pytest.approx(sum(result), abs=1e-6) == 0.0

    def test_ordering_preserved(self):
        """Higher reward → higher advantage."""
        rewards = [0.1, 0.9, 0.5]
        result = compute_group_advantages(rewards)
        assert result[1] > result[2] > result[0]

    def test_known_values(self):
        rewards = [1.0, 2.0, 3.0, 4.0]
        # mean=2.5, var=1.25, std=sqrt(1.25)≈1.1180
        result = compute_group_advantages(rewards)
        std = 1.25**0.5
        expected = [(r - 2.5) / std for r in rewards]
        for r, e in zip(result, expected):
            assert pytest.approx(r, abs=1e-6) == e


try:
    import torch as _torch  # noqa: F401
    _has_torch = True
except ImportError:
    _has_torch = False

requires_torch = pytest.mark.skipif(not _has_torch, reason="torch not installed")


@requires_torch
class TestClippedSurrogateLoss:
    def test_no_clipping_when_ratio_near_one(self):
        import torch
        from src.training.grpo import clipped_surrogate_loss

        ratios = torch.ones(4)
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
        loss = clipped_surrogate_loss(ratios, advantages, clip_eps=0.2)
        # With ratio=1, loss = -mean(advantages) = 0.0
        assert pytest.approx(loss.item(), abs=1e-5) == 0.0

    def test_clipping_limits_large_ratios(self):
        import torch
        from src.training.grpo import clipped_surrogate_loss

        ratios = torch.tensor([2.0, 2.0])
        advantages = torch.tensor([1.0, 1.0])
        loss = clipped_surrogate_loss(ratios, advantages, clip_eps=0.2)
        # Clipped to 1.2, so loss = -1.2
        assert pytest.approx(loss.item(), abs=1e-5) == -1.2


@requires_torch
class TestKLPenalty:
    def test_zero_kl_when_same(self):
        import torch
        from src.training.grpo import kl_penalty

        log_probs = torch.tensor([-1.0, -2.0])
        result = kl_penalty(log_probs, log_probs, beta=0.1)
        assert pytest.approx(result.item(), abs=1e-6) == 0.0

    def test_positive_kl(self):
        import torch
        from src.training.grpo import kl_penalty

        log_probs = torch.tensor([-0.5])
        ref_log_probs = torch.tensor([-1.0])
        result = kl_penalty(log_probs, ref_log_probs, beta=0.1)
        # 0.1 * mean(-0.5 - (-1.0)) = 0.1 * 0.5 = 0.05
        assert pytest.approx(result.item(), abs=1e-6) == 0.05


@requires_torch
class TestAsymmetricClippedSurrogateLoss:
    def test_asymmetric_bounds(self):
        import torch
        from src.training.grpo import asymmetric_clipped_surrogate_loss

        ratios = torch.tensor([2.0, 2.0])
        advantages = torch.tensor([1.0, 1.0])
        loss = asymmetric_clipped_surrogate_loss(
            ratios, advantages, clip_eps=0.2, clip_eps_high=0.28
        )
        # Clipped to 1.28, so loss = -1.28
        assert pytest.approx(loss.item(), abs=1e-5) == -1.28

    def test_low_clip_matches_symmetric(self):
        import torch
        from src.training.grpo import asymmetric_clipped_surrogate_loss

        ratios = torch.tensor([0.5, 0.5])
        advantages = torch.tensor([1.0, 1.0])
        loss = asymmetric_clipped_surrogate_loss(
            ratios, advantages, clip_eps=0.2, clip_eps_high=0.28
        )
        # Clipped to 0.8 (low bound), so loss = -0.5 (min of 0.5*1 and 0.8*1)
        assert pytest.approx(loss.item(), abs=1e-5) == -0.5

    def test_no_clipping_when_ratio_near_one(self):
        import torch
        from src.training.grpo import asymmetric_clipped_surrogate_loss

        ratios = torch.ones(4)
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
        loss = asymmetric_clipped_surrogate_loss(ratios, advantages)
        assert pytest.approx(loss.item(), abs=1e-5) == 0.0


@requires_torch
class TestCombinedLoss:
    def test_weighted_combination(self):
        import torch
        from src.training.grpo import combined_loss

        rl = torch.tensor(0.5)
        opd = torch.tensor(0.3)
        result = combined_loss(rl, opd, w_rl=0.7, w_opd=0.3)
        # 0.7 * 0.5 + 0.3 * 0.3 = 0.35 + 0.09 = 0.44
        assert pytest.approx(result.item(), abs=1e-5) == 0.44

    def test_pure_rl(self):
        import torch
        from src.training.grpo import combined_loss

        rl = torch.tensor(1.0)
        opd = torch.tensor(0.0)
        result = combined_loss(rl, opd, w_rl=1.0, w_opd=0.0)
        assert pytest.approx(result.item(), abs=1e-5) == 1.0
