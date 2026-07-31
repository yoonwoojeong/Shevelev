"""EDM (Elucidating the Design Space of Diffusion-Based Generative Models) configuration.

Implements the full Karras et al. continuous-time preconditioning framework where
σ(t) = t (identity mapping — time IS noise level).  All preconditioning scalars
are derived from a single hyper-parameter σ_data.

Reference:  Karras et al., "Elucidating the Design Space of Diffusion-Based
Generative Models", NeurIPS 2022.
"""

from __future__ import annotations

import dataclasses
import math

import torch


@dataclasses.dataclass
class EDMConfig:
    """Central configuration for the EDM continuous-time diffusion model.

    Holds every hyper-parameter needed for training, sampling, and
    architecture construction.  Static helper methods compute the
    preconditioning scalars and the Karras noise schedule so that
    the same math is shared across training and inference code.
    """

    # ------------------------------------------------------------------ #
    #  Data
    # ------------------------------------------------------------------ #
    img_resolution: int = 32
    """Spatial resolution of generated images (pixels)."""

    img_channels: int = 3
    """Number of image colour channels (3 for RGB)."""

    # ------------------------------------------------------------------ #
    #  EDM preconditioning
    # ------------------------------------------------------------------ #
    sigma_data: float = 0.5
    """Assumed standard deviation of the data distribution.

    Controls how the skip connection and output scaling balance the
    raw-network prediction against the noisy input.
    """

    # ------------------------------------------------------------------ #
    #  Training noise distribution  (log-normal)
    # ------------------------------------------------------------------ #
    P_mean: float = -1.2
    """Mean of ln(σ) during training noise sampling."""

    P_std: float = 1.2
    """Std-dev of ln(σ) during training noise sampling."""

    # ------------------------------------------------------------------ #
    #  Sampling schedule
    # ------------------------------------------------------------------ #
    sigma_min: float = 0.002
    """Smallest noise level in the Karras schedule."""

    sigma_max: float = 80.0
    """Largest noise level in the Karras schedule."""

    rho: float = 7.0
    """Karras schedule exponent (controls spacing of σ_i)."""

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    learning_rate: float = 2e-4
    """Adam learning rate."""

    batch_size: int = 128
    """Mini-batch size."""

    ema_decay: float = 0.9999
    """Exponential-moving-average decay for the model weights."""

    total_steps: int = 100_000
    """Total number of training gradient steps."""

    # ------------------------------------------------------------------ #
    #  Architecture
    # ------------------------------------------------------------------ #
    model_channels: int = 128
    """Base channel count of the U-Net backbone."""

    channel_mult: tuple[int, ...] = (1, 2, 2)
    """Per-resolution channel multipliers (len = number of down-sample stages + 1)."""

    time_embed_dim: int = 256
    """Dimensionality of the time-embedding vector injected into the backbone."""

    # ================================================================== #
    #  Preconditioning scalars  (static, operate on tensors)
    # ================================================================== #

    @staticmethod
    def c_skip(sigma: torch.Tensor, sigma_data: float = 0.5) -> torch.Tensor:
        """Skip-connection coefficient.

        c_skip(σ) = σ_data² / (σ² + σ_data²)

        Blends the noisy input *x* directly into the denoiser output so that
        the backbone only has to predict the residual.

        Args:
            sigma: Noise level tensor of arbitrary shape.
            sigma_data: Assumed data standard deviation.

        Returns:
            Tensor of the same shape as *sigma*.
        """
        return sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)

    @staticmethod
    def c_out(sigma: torch.Tensor, sigma_data: float = 0.5) -> torch.Tensor:
        """Output scaling coefficient.

        c_out(σ) = σ · σ_data / √(σ² + σ_data²)

        Scales the raw backbone output before it is added to the skip path.

        Args:
            sigma: Noise level tensor of arbitrary shape.
            sigma_data: Assumed data standard deviation.

        Returns:
            Tensor of the same shape as *sigma*.
        """
        return sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()

    @staticmethod
    def c_in(sigma: torch.Tensor, sigma_data: float = 0.5) -> torch.Tensor:
        """Input scaling coefficient.

        c_in(σ) = 1 / √(σ² + σ_data²)

        Normalises the noisy input so that the backbone always sees
        unit-variance data.

        Args:
            sigma: Noise level tensor of arbitrary shape.
            sigma_data: Assumed data standard deviation.

        Returns:
            Tensor of the same shape as *sigma*.
        """
        return 1.0 / (sigma ** 2 + sigma_data ** 2).sqrt()

    @staticmethod
    def c_noise(sigma: torch.Tensor) -> torch.Tensor:
        """Noise conditioning value fed to the backbone.

        c_noise(σ) = (1/4) · ln(σ)

        Maps the noise level to a bounded, slowly varying scalar that is
        easier for the network to consume than raw σ.

        Args:
            sigma: Noise level tensor (must be > 0).

        Returns:
            Tensor of the same shape as *sigma*.
        """
        return 0.25 * sigma.log()

    @staticmethod
    def loss_weight(sigma: torch.Tensor, sigma_data: float = 0.5) -> torch.Tensor:
        """Per-sample loss weighting λ(σ).

        λ(σ) = (σ² + σ_data²) / (σ² · σ_data²)

        This is the *inverse* of c_out², ensuring that the weighted loss
        is equivalent to minimising the raw-backbone MSE (Table 1 of Karras).

        Args:
            sigma: Noise level tensor of arbitrary shape.
            sigma_data: Assumed data standard deviation.

        Returns:
            Tensor of the same shape as *sigma*.
        """
        return (sigma ** 2 + sigma_data ** 2) / (sigma ** 2 * sigma_data ** 2)

    # ================================================================== #
    #  Karras noise schedule
    # ================================================================== #

    @staticmethod
    def get_karras_sigmas(
        n_steps: int,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
    ) -> torch.Tensor:
        """Build the Karras deterministic noise schedule.

        σ_i = (σ_min^{1/ρ} + i/(N-1) · (σ_max^{1/ρ} - σ_min^{1/ρ}))^ρ

        for i = 0, 1, …, N-1.  An extra σ_{N} = 0 is appended so that the
        final ODE step lands exactly on the data manifold.

        Args:
            n_steps: Number of discretisation steps *N* (the returned tensor
                     has length N + 1 because of the trailing zero).
            sigma_min: Smallest noise level.
            sigma_max: Largest noise level.
            rho: Schedule exponent controlling step spacing.

        Returns:
            1-D float tensor of shape ``(n_steps + 1,)`` with noise levels
            arranged in **descending** order, ending with 0.
        """
        ramp = torch.linspace(0, 1, n_steps)                        # i / (N-1)
        min_inv_rho = sigma_min ** (1.0 / rho)
        max_inv_rho = sigma_max ** (1.0 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat([sigmas, sigmas.new_zeros(1)])              # append σ=0
