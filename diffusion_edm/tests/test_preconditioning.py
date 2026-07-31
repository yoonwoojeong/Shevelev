"""
test_preconditioning.py — Verify EDM preconditioning numerical safety.

Tests that the preconditioning scaling functions (c_skip, c_out, c_in, c_noise)
produce finite values across the entire operating range σ ∈ [1e-6, 1e3], and
that the full forward pass through EDMPrecond returns no NaN/Inf for extreme σ.
"""

import sys
import os
import math

import torch
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EDMConfig
from model_wrapper import EDMPrecond


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config() -> EDMConfig:
    """Create a lightweight config for testing."""
    return EDMConfig(
        img_resolution=32,
        img_channels=3,
        model_channels=32,          # Small for fast tests
        channel_mult=(1, 2),        # Only 2 levels for speed
        time_embed_dim=64,
    )


@pytest.fixture(scope="module")
def model(config: EDMConfig) -> EDMPrecond:
    """Create a small EDMPrecond model for testing."""
    m = EDMPrecond(config)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Preconditioning Function Tests
# ---------------------------------------------------------------------------

class TestScalingFunctions:
    """Verify c_skip, c_out, c_in, c_noise over a wide range of σ."""

    SIGMA_RANGE = [1e-6, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 80.0, 1000.0]
    SIGMA_DATA = 0.5

    def test_c_skip_finite(self) -> None:
        """c_skip(σ) must be finite for all σ in operating range."""
        for s in self.SIGMA_RANGE:
            sigma = torch.tensor([s])
            result = EDMConfig.c_skip(sigma, self.SIGMA_DATA)
            assert torch.isfinite(result).all(), f"c_skip NaN/Inf at σ={s}"

    def test_c_out_finite(self) -> None:
        """c_out(σ) must be finite for all σ in operating range."""
        for s in self.SIGMA_RANGE:
            sigma = torch.tensor([s])
            result = EDMConfig.c_out(sigma, self.SIGMA_DATA)
            assert torch.isfinite(result).all(), f"c_out NaN/Inf at σ={s}"

    def test_c_in_finite(self) -> None:
        """c_in(σ) must be finite for all σ in operating range."""
        for s in self.SIGMA_RANGE:
            sigma = torch.tensor([s])
            result = EDMConfig.c_in(sigma, self.SIGMA_DATA)
            assert torch.isfinite(result).all(), f"c_in NaN/Inf at σ={s}"

    def test_c_noise_finite(self) -> None:
        """c_noise(σ) must be finite for positive σ."""
        for s in self.SIGMA_RANGE:
            if s > 0:
                sigma = torch.tensor([s])
                result = EDMConfig.c_noise(sigma)
                assert torch.isfinite(result).all(), f"c_noise NaN/Inf at σ={s}"

    def test_c_skip_range(self) -> None:
        """c_skip(σ) ∈ (0, 1] for all positive σ."""
        sigma = torch.tensor(self.SIGMA_RANGE)
        result = EDMConfig.c_skip(sigma, self.SIGMA_DATA)
        assert (result > 0).all(), "c_skip must be positive"
        assert (result <= 1.0 + 1e-6).all(), "c_skip must be ≤ 1"

    def test_c_in_normalizes_to_unit_variance(self) -> None:
        """c_in(σ) should normalize (σ² + σ_data²) to 1.

        Var(c_in · x_t) = c_in² · (σ² + σ_data²) should ≈ 1.
        """
        for s in self.SIGMA_RANGE:
            sigma = torch.tensor([s])
            c_in = EDMConfig.c_in(sigma, self.SIGMA_DATA)
            normalized_var = (c_in ** 2) * (s ** 2 + self.SIGMA_DATA ** 2)
            assert abs(normalized_var.item() - 1.0) < 1e-5, (
                f"c_in failed unit-variance normalization at σ={s}: got {normalized_var.item()}"
            )


# ---------------------------------------------------------------------------
# Full Forward Pass Tests
# ---------------------------------------------------------------------------

class TestEDMPrecondForward:
    """Verify the full preconditioned forward pass handles extreme σ values."""

    EXTREME_SIGMAS = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 80.0]

    @torch.no_grad()
    def test_no_nan_across_sigma_range(self, model: EDMPrecond) -> None:
        """Forward pass must produce finite outputs for all σ in range."""
        x = torch.randn(2, 3, 32, 32)
        for s in self.EXTREME_SIGMAS:
            sigma = torch.full((2,), s)
            output = model(x, sigma)
            assert torch.isfinite(output).all(), (
                f"EDMPrecond output contains NaN/Inf at σ={s}"
            )

    @torch.no_grad()
    def test_output_shape(self, model: EDMPrecond) -> None:
        """Output shape must match input image shape."""
        x = torch.randn(4, 3, 32, 32)
        sigma = torch.full((4,), 1.0)
        output = model(x, sigma)
        assert output.shape == x.shape, (
            f"Shape mismatch: input {x.shape} vs output {output.shape}"
        )

    @torch.no_grad()
    def test_near_zero_sigma_safety(self, model: EDMPrecond) -> None:
        """σ very close to 0 must NOT produce NaN (clamping must work)."""
        x = torch.randn(2, 3, 32, 32)
        sigma = torch.full((2,), 1e-6)
        output = model(x, sigma)
        assert torch.isfinite(output).all(), (
            "EDMPrecond produced NaN/Inf for σ ≈ 0 (numerical singularity)"
        )

    @torch.no_grad()
    def test_skip_connection_dominates_at_low_sigma(self, model: EDMPrecond) -> None:
        """As σ → 0, c_skip → 1 and c_out → 0, so output ≈ input."""
        x = torch.randn(2, 3, 32, 32)
        sigma = torch.full((2,), 1e-4)
        output = model(x, sigma)
        # c_skip at σ=1e-4 with σ_data=0.5: ≈ 0.9999999..., very close to 1
        # c_out at σ=1e-4: ≈ 1e-4, very small
        # So output ≈ x within tolerance
        relative_error = (output - x).abs().mean() / x.abs().mean()
        assert relative_error < 0.5, (
            f"At σ≈0, output should be close to input (skip-dominated). "
            f"Relative error: {relative_error:.4f}"
        )

    @torch.no_grad()
    def test_loss_weight_finite(self, model: EDMPrecond) -> None:
        """Loss weight λ(σ) must be finite for all operating σ values."""
        for s in self.EXTREME_SIGMAS:
            sigma = torch.tensor([s])
            weight = model.get_loss_weight(sigma)
            assert torch.isfinite(weight).all(), (
                f"Loss weight NaN/Inf at σ={s}"
            )
            assert (weight > 0).all(), (
                f"Loss weight must be positive, got {weight.item()} at σ={s}"
            )


# ---------------------------------------------------------------------------
# Gradient Flow Test
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Verify gradients flow properly through the preconditioned model."""

    def test_gradients_not_nan(self, model: EDMPrecond, config: EDMConfig) -> None:
        """Backpropagation must not produce NaN gradients."""
        model_copy = EDMPrecond(config)
        model_copy.train()

        x = torch.randn(2, 3, 32, 32)
        sigma = torch.full((2,), 1.0)
        target = torch.randn_like(x)

        output = model_copy(x, sigma)
        loss = (output - target).pow(2).mean()
        loss.backward()

        for name, param in model_copy.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"NaN gradient in parameter: {name}"
                )
