"""EDM preconditioning wrapper.

Implements the full denoiser  D_θ(x_t, σ)  from
*Elucidating the Design Space of Diffusion-Based Generative Models*
(Karras et al., 2022), Table 1 / VP column with σ(t)=t.

The wrapper applies analytic skip / output / input scalings and delegates
all learned computation to an inner UNet backbone.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

try:
    from .backbone import UNet
    from .config import EDMConfig
except ImportError:
    from backbone import UNet
    from config import EDMConfig


class EDMPrecond(nn.Module):
    """Karras‑EDM preconditioning wrapper around a UNet backbone.

    Given noisy input  x_t  and continuous noise level  σ  the denoiser is

        D_θ(x_t, σ) = c_skip(σ) · x_t  +  c_out(σ) · F_θ(c_in(σ) · x_t, σ)

    where F_θ is the raw backbone network and the scaling coefficients are

        c_skip(σ) = σ_data² / (σ² + σ_data²)
        c_out(σ)  = σ · σ_data / √(σ² + σ_data²)
        c_in(σ)   = 1 / √(σ² + σ_data²)
        c_noise(σ)= ¼ ln σ        (applied *inside* the backbone's FourierFeatures)

    The backbone receives **raw σ** — its internal FourierFeatures layer is
    responsible for computing c_noise.

    Parameters
    ----------
    config : EDMConfig
        Dataclass / namespace carrying all hyper‑parameters.  Must expose at
        least ``sigma_data``, ``img_channels``, ``img_resolution``,
        ``model_channels``, and ``channel_mult``.
    """

    # Minimum sigma for numerical safety (avoids log(0) and division by 0).
    _SIGMA_MIN_CLAMP: float = 1e-4

    def __init__(self, config: EDMConfig) -> None:
        super().__init__()
        self.sigma_data: float = config.sigma_data
        self.backbone = UNet(
            img_channels=config.img_channels,
            model_channels=config.model_channels,
            channel_mult=config.channel_mult,
            time_embed_dim=config.time_embed_dim,
        )

    # ------------------------------------------------------------------
    # Scaling coefficients
    # ------------------------------------------------------------------

    def _reshape_sigma(self, sigma: Tensor) -> Tensor:
        """Reshape sigma from (B,) to (B, 1, 1, 1) for spatial broadcasting."""
        return sigma.view(-1, 1, 1, 1)

    def c_skip(self, sigma: Tensor) -> Tensor:
        r"""Skip‑connection coefficient:  σ_data² / (σ² + σ_data²)."""
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: Tensor) -> Tensor:
        r"""Output scaling:  σ · σ_data / √(σ² + σ_data²)."""
        return sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def c_in(self, sigma: Tensor) -> Tensor:
        r"""Input scaling:  1 / √(σ² + σ_data²)."""
        return 1.0 / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    # ------------------------------------------------------------------
    # Forward pass — the full denoiser D_θ
    # ------------------------------------------------------------------

    def forward(self, x_t: Tensor, sigma: Tensor) -> Tensor:
        """Evaluate the preconditioned denoiser D_θ(x_t, σ).

        Parameters
        ----------
        x_t : Tensor
            Noisy images of shape ``(B, C, H, W)``.
        sigma : Tensor
            Per‑sample continuous noise levels of shape ``(B,)``.
            These are **raw** float values, not pre‑processed.

        Returns
        -------
        Tensor
            Denoised prediction of shape ``(B, C, H, W)``.
        """
        # Numerical safety clamp.
        sigma = sigma.clamp(min=self._SIGMA_MIN_CLAMP)

        # Reshape for spatial broadcasting: (B,) → (B, 1, 1, 1).
        sigma_4d = self._reshape_sigma(sigma)

        # Preconditioning coefficients.
        c_skip = self.c_skip(sigma_4d)   # (B, 1, 1, 1)
        c_out = self.c_out(sigma_4d)     # (B, 1, 1, 1)
        c_in = self.c_in(sigma_4d)       # (B, 1, 1, 1)

        # Backbone evaluation — receives raw sigma (1‑D, shape (B,)).
        # The backbone's FourierFeatures layer internally computes c_noise.
        f_theta = self.backbone(c_in * x_t, sigma)

        # Full denoiser.
        denoised = c_skip * x_t + c_out * f_theta
        return denoised

    # ------------------------------------------------------------------
    # Auxiliary helpers
    # ------------------------------------------------------------------

    def get_loss_weight(self, sigma: Tensor) -> Tensor:
        r"""Per‑sample loss weighting  λ(σ) = (σ² + σ_data²) / (σ² · σ_data²).

        Parameters
        ----------
        sigma : Tensor
            Noise levels of arbitrary shape (typically ``(B,)``).

        Returns
        -------
        Tensor
            Loss weights, same shape as *sigma*.
        """
        return (sigma ** 2 + self.sigma_data ** 2) / (
            sigma ** 2 * self.sigma_data ** 2
        )

    def get_score(
        self, x_t: Tensor, sigma: Tensor, denoised: Tensor
    ) -> Tensor:
        r"""Compute the Stein score  ∇_x log p(x_t | σ).

        Under the EDM formulation with Gaussian perturbation kernel:

            score = (denoised − x_t) / σ²

        Parameters
        ----------
        x_t : Tensor
            Noisy images ``(B, C, H, W)``.
        sigma : Tensor
            Noise levels ``(B,)``.
        denoised : Tensor
            Denoised prediction ``(B, C, H, W)`` — i.e. output of
            :meth:`forward`.

        Returns
        -------
        Tensor
            Score estimate ``(B, C, H, W)``.
        """
        sigma_4d = self._reshape_sigma(sigma)
        return (denoised - x_t) / sigma_4d ** 2
