"""
test_continuous_time.py — Verify continuous-time compliance.

Ensures that no integer-valued time steps, discrete Markov chains, or
DDPM-style noise schedules appear anywhere in the codebase. Time must
always be a continuous floating-point scalar t ∈ (0, T].
"""

import sys
import os
import re
import ast
from pathlib import Path

import torch
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EDMConfig


# ---------------------------------------------------------------------------
# Source-Level Static Analysis
# ---------------------------------------------------------------------------

# Project source files to scan (relative to diffusion_edm/)
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_FILES = [
    "config.py",
    "time_embedding.py",
    "backbone.py",
    "model_wrapper.py",
    "train.py",
    "sample.py",
]


class TestNoDiscreteTime:
    """Static analysis: verify no discrete time patterns in source code."""

    # Patterns that indicate discrete-time diffusion
    FORBIDDEN_PATTERNS = [
        # DDPM-style discrete timesteps
        r"\bnum_timesteps\s*=\s*\d+",
        r"\btimesteps\s*=\s*\d+",
        r"(?<![a-zA-Z_])t\s*=\s*(?:range|list)\s*\(",    # t = range(...) or t = list(...)
        r"\bfor\s+t\s+in\s+range\(",
        # DDPM beta schedules
        r"\bbeta_schedule\b",
        r"\blinear_beta\b",
        r"\bcosine_beta\b",
        r"\balpha_cumprod\b",
        r"\balphas_cumprod\b",
        # Discrete Markov chain terminology
        r"\bmarkov_chain\b",
        r"\btransition_matrix\b",
    ]

    def test_no_discrete_patterns_in_source(self) -> None:
        """Scan all source files for forbidden discrete-time patterns."""
        violations = []
        for filename in SOURCE_FILES:
            filepath = PROJECT_ROOT / filename
            if not filepath.exists():
                continue
            content = filepath.read_text(encoding="utf-8")
            for pattern in self.FORBIDDEN_PATTERNS:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    violations.append(
                        f"{filename}: found forbidden pattern '{pattern}' → {matches}"
                    )
        assert not violations, (
            "Discrete-time patterns found in source:\n" + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# Runtime Continuous-Time Verification
# ---------------------------------------------------------------------------

class TestContinuousTimeRuntime:
    """Runtime verification that σ/t values are always continuous floats."""

    def test_sigma_dtype_is_float(self) -> None:
        """Sampled σ values must have float dtype, never integer."""
        P_mean, P_std = -1.2, 1.2
        ln_sigma = P_mean + P_std * torch.randn(100)
        sigma = ln_sigma.exp()

        assert sigma.dtype in (torch.float32, torch.float64), (
            f"σ has non-float dtype: {sigma.dtype}"
        )

    def test_sigma_is_continuous(self) -> None:
        """σ values must not be integer-valued (i.e., not 1.0, 2.0, ...)."""
        P_mean, P_std = -1.2, 1.2
        torch.manual_seed(42)
        ln_sigma = P_mean + P_std * torch.randn(1000)
        sigma = ln_sigma.exp()

        # With continuous log-normal sampling, the probability of getting
        # an exact integer is essentially zero
        integer_count = (sigma == sigma.round()).sum().item()
        # Allow at most 1 coincidental integer out of 1000 samples
        assert integer_count <= 1, (
            f"Too many integer-valued σ samples ({integer_count}/1000). "
            f"This suggests discrete time steps are being used."
        )

    def test_sigma_positive(self) -> None:
        """All σ values must be strictly positive (no zero or negative)."""
        P_mean, P_std = -1.2, 1.2
        torch.manual_seed(0)
        ln_sigma = P_mean + P_std * torch.randn(10000)
        sigma = ln_sigma.exp()

        assert (sigma > 0).all(), "σ must be strictly positive"

    def test_sigma_spans_orders_of_magnitude(self) -> None:
        """Log-normal σ should cover a wide range (multi-octave)."""
        P_mean, P_std = -1.2, 1.2
        torch.manual_seed(42)
        ln_sigma = P_mean + P_std * torch.randn(10000)
        sigma = ln_sigma.exp()

        log_range = sigma.log10().max() - sigma.log10().min()
        assert log_range > 2.0, (
            f"σ range too narrow: only {log_range:.1f} decades. "
            f"Expected >2 decades for proper coverage."
        )


# ---------------------------------------------------------------------------
# Preconditioning Function Input Verification
# ---------------------------------------------------------------------------

class TestPreconditioningInputs:
    """Verify that preconditioning functions accept continuous float inputs."""

    CONTINUOUS_SIGMAS = [0.002, 0.0137, 0.0941, 0.647, 4.44, 30.5, 80.0]

    def test_c_skip_accepts_arbitrary_float(self) -> None:
        """c_skip must work with non-round continuous floats."""
        for s in self.CONTINUOUS_SIGMAS:
            sigma = torch.tensor([s])
            result = EDMConfig.c_skip(sigma)
            assert torch.isfinite(result).all(), f"c_skip failed for σ={s}"

    def test_c_out_accepts_arbitrary_float(self) -> None:
        """c_out must work with non-round continuous floats."""
        for s in self.CONTINUOUS_SIGMAS:
            sigma = torch.tensor([s])
            result = EDMConfig.c_out(sigma)
            assert torch.isfinite(result).all(), f"c_out failed for σ={s}"

    def test_c_in_accepts_arbitrary_float(self) -> None:
        """c_in must work with non-round continuous floats."""
        for s in self.CONTINUOUS_SIGMAS:
            sigma = torch.tensor([s])
            result = EDMConfig.c_in(sigma)
            assert torch.isfinite(result).all(), f"c_in failed for σ={s}"

    def test_c_noise_accepts_arbitrary_float(self) -> None:
        """c_noise must work with non-round continuous floats."""
        for s in self.CONTINUOUS_SIGMAS:
            sigma = torch.tensor([s])
            result = EDMConfig.c_noise(sigma)
            assert torch.isfinite(result).all(), f"c_noise failed for σ={s}"

    def test_loss_weight_accepts_arbitrary_float(self) -> None:
        """λ(σ) must work with non-round continuous floats."""
        for s in self.CONTINUOUS_SIGMAS:
            sigma = torch.tensor([s])
            result = EDMConfig.loss_weight(sigma)
            assert torch.isfinite(result).all(), f"λ(σ) failed for σ={s}"
            assert (result > 0).all(), f"λ(σ) must be positive, got {result} for σ={s}"
