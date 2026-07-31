"""
test_determinism.py — Verify ODE sampler determinism.

Tests that the Probability Flow ODE sampler produces exactly identical
outputs when given the same initial noise vector, confirming that no
stochastic noise is injected during the reverse integration.
"""

import sys
import os

import torch
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EDMConfig
from model_wrapper import EDMPrecond
from sample import ODESampler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config() -> EDMConfig:
    """Create a lightweight config for testing."""
    return EDMConfig(
        img_resolution=32,
        img_channels=3,
        model_channels=32,
        channel_mult=(1, 2),
        time_embed_dim=64,
    )


@pytest.fixture(scope="module")
def sampler(config: EDMConfig) -> ODESampler:
    """Create a sampler with a small model for testing."""
    model = EDMPrecond(config)
    model.eval()
    return ODESampler(model, config)


# ---------------------------------------------------------------------------
# Determinism Tests
# ---------------------------------------------------------------------------

class TestDeterministicSampling:
    """Verify that the ODE sampler is fully deterministic."""

    @torch.no_grad()
    def test_euler_identical_outputs(self, sampler: ODESampler) -> None:
        """Two Euler samples with same seed must be bit-for-bit identical."""
        out1 = sampler.sample(num_samples=2, num_steps=5, solver="euler", seed=42)
        out2 = sampler.sample(num_samples=2, num_steps=5, solver="euler", seed=42)
        l2_distance = (out1 - out2).pow(2).sum().sqrt().item()
        assert l2_distance == 0.0, (
            f"Euler sampler is non-deterministic! L2 distance = {l2_distance}"
        )

    @torch.no_grad()
    def test_heun_identical_outputs(self, sampler: ODESampler) -> None:
        """Two Heun samples with same seed must be bit-for-bit identical."""
        out1 = sampler.sample(num_samples=2, num_steps=5, solver="heun", seed=42)
        out2 = sampler.sample(num_samples=2, num_steps=5, solver="heun", seed=42)
        l2_distance = (out1 - out2).pow(2).sum().sqrt().item()
        assert l2_distance == 0.0, (
            f"Heun sampler is non-deterministic! L2 distance = {l2_distance}"
        )

    @torch.no_grad()
    def test_different_seeds_differ(self, sampler: ODESampler) -> None:
        """Different seeds must produce different outputs."""
        out1 = sampler.sample(num_samples=2, num_steps=5, solver="euler", seed=42)
        out2 = sampler.sample(num_samples=2, num_steps=5, solver="euler", seed=123)
        l2_distance = (out1 - out2).pow(2).sum().sqrt().item()
        assert l2_distance > 0.0, "Different seeds produced identical outputs"

    @torch.no_grad()
    def test_output_shape(self, sampler: ODESampler, config: EDMConfig) -> None:
        """Sampler output shape must match (N, C, H, W)."""
        num_samples = 4
        out = sampler.sample(num_samples=num_samples, num_steps=3, solver="euler", seed=0)
        expected = (num_samples, config.img_channels, config.img_resolution, config.img_resolution)
        assert out.shape == expected, f"Expected shape {expected}, got {out.shape}"


# ---------------------------------------------------------------------------
# No-Stochasticity-in-Loop Tests
# ---------------------------------------------------------------------------

class TestNoStochasticNoise:
    """Verify that torch.randn is NOT called inside the ODE integration loop."""

    @torch.no_grad()
    def test_no_randn_in_euler_loop(self, sampler: ODESampler) -> None:
        """The Euler solver must not call torch.randn internally.

        We verify this indirectly: if the solver is deterministic given
        the same initial z, it cannot be injecting random noise.
        """
        # Create identical starting tensors
        torch.manual_seed(99)
        z = torch.randn(1, 3, 32, 32) * sampler.config.sigma_max
        sigmas = sampler.get_karras_schedule(5)

        out1 = sampler.sample_euler(z.clone(), sigmas)
        out2 = sampler.sample_euler(z.clone(), sigmas)

        l2 = (out1 - out2).pow(2).sum().sqrt().item()
        assert l2 == 0.0, f"Euler loop appears stochastic (L2 = {l2})"

    @torch.no_grad()
    def test_no_randn_in_heun_loop(self, sampler: ODESampler) -> None:
        """The Heun solver must not call torch.randn internally."""
        torch.manual_seed(99)
        z = torch.randn(1, 3, 32, 32) * sampler.config.sigma_max
        sigmas = sampler.get_karras_schedule(5)

        out1 = sampler.sample_heun(z.clone(), sigmas)
        out2 = sampler.sample_heun(z.clone(), sigmas)

        l2 = (out1 - out2).pow(2).sum().sqrt().item()
        assert l2 == 0.0, f"Heun loop appears stochastic (L2 = {l2})"


# ---------------------------------------------------------------------------
# Karras Schedule Tests
# ---------------------------------------------------------------------------

class TestKarrasSchedule:
    """Verify the Karras noise schedule is correctly computed."""

    def test_schedule_monotonic_decreasing(self, sampler: ODESampler) -> None:
        """Sigmas must be strictly decreasing from σ_max to 0."""
        sigmas = sampler.get_karras_schedule(20)
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i + 1], (
                f"Schedule not decreasing at index {i}: "
                f"σ[{i}]={sigmas[i]:.6f} ≤ σ[{i+1}]={sigmas[i+1]:.6f}"
            )

    def test_schedule_endpoints(self, sampler: ODESampler) -> None:
        """Schedule must start at σ_max and end at 0."""
        sigmas = sampler.get_karras_schedule(20)
        assert abs(sigmas[0].item() - sampler.config.sigma_max) < 1e-4, (
            f"First sigma should be σ_max={sampler.config.sigma_max}, got {sigmas[0]}"
        )
        assert sigmas[-1].item() == 0.0, (
            f"Last sigma should be 0, got {sigmas[-1]}"
        )

    def test_schedule_all_continuous_floats(self, sampler: ODESampler) -> None:
        """All schedule values must be continuous floats, not integers."""
        sigmas = sampler.get_karras_schedule(20)
        assert sigmas.dtype in (torch.float32, torch.float64), (
            f"Schedule dtype must be float, got {sigmas.dtype}"
        )
