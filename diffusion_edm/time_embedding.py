"""Continuous-time injection modules for the EDM diffusion backbone.

Provides two building blocks that let the U-Net condition on the
continuous noise level σ:

* **FourierFeatures** – maps a scalar σ (per sample) to a dense vector
  via  c_noise(σ) → sinusoidal positional encoding → MLP.
* **AdaptiveLayerNorm** (AdaLN) – modulates normalised spatial feature
  maps using scale/shift parameters derived from the time embedding.

Both modules are self-contained and import only PyTorch + the project
config for the ``c_noise`` transform.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from .config import EDMConfig
except ImportError:
    from config import EDMConfig


# ====================================================================== #
#  Fourier Features  (σ → embedding vector)
# ====================================================================== #

class FourierFeatures(nn.Module):
    """Map a continuous noise level σ to a dense time-embedding vector.

    Pipeline (per sample)::

        σ  ──►  c_noise(σ) = 0.25·ln(σ)
           ──►  sinusoidal positional encoding  (fixed geometric freqs)
           ──►  Linear  ──►  SiLU  ──►  Linear
           ──►  e ∈ ℝ^{time_embed_dim}

    The sinusoidal encoding uses **fixed** (non-learnable) frequencies
    arranged in a geometric progression, identical to the classic
    Transformer positional encoding but applied to a single scalar.

    Args:
        n_frequencies: Number of frequency bands.  The positional encoding
                       has dimension ``2 * n_frequencies`` (sin + cos).
        time_embed_dim: Final output dimensionality of the embedding.
    """

    def __init__(self, n_frequencies: int = 128, time_embed_dim: int = 256) -> None:
        super().__init__()
        self.n_frequencies = n_frequencies
        self.time_embed_dim = time_embed_dim

        # Fixed geometric frequencies: ω_k = exp(-ln(10000) · k / (D-1))
        # where k = 0, 1, …, D-1  and D = n_frequencies.
        # We store them as a non-learnable buffer so they move with the
        # model to the correct device / dtype automatically.
        freqs = torch.exp(
            -math.log(10_000.0)
            * torch.arange(n_frequencies, dtype=torch.float32)
            / max(n_frequencies - 1, 1)
        )
        self.register_buffer("freqs", freqs)                       # (D,)

        # Two-layer MLP: (2D) → time_embed_dim → time_embed_dim
        encoding_dim = 2 * n_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(encoding_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    # ------------------------------------------------------------------ #

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the time embedding for a batch of noise levels.

        Args:
            sigma: Noise level tensor of shape ``(B,)`` with **positive**
                   continuous float values (never integer time-step indices).

        Returns:
            Time embedding of shape ``(B, time_embed_dim)``.
        """
        # 1. Map σ → c_noise(σ) = 0.25 · ln(σ)   — bounded conditioning
        c = EDMConfig.c_noise(sigma)                               # (B,)

        # 2. Sinusoidal positional encoding with fixed geometric freqs
        #    angles = c[:, None] * freqs[None, :]   →  (B, D)
        angles = c.unsqueeze(1) * self.freqs.unsqueeze(0)          # (B, D)
        encoding = torch.cat([angles.sin(), angles.cos()], dim=1)  # (B, 2D)

        # 3. MLP projection → final embedding
        return self.mlp(encoding)                                  # (B, E)


# ====================================================================== #
#  Adaptive Layer Norm  (AdaLN)
# ====================================================================== #

class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalisation (AdaLN) conditioned on a time embedding.

    Given spatial features **h** ∈ ℝ^{B×C×H×W} and a time embedding
    **e** ∈ ℝ^{B×E}, the module:

    1. Projects **e** through  Linear → SiLU → Linear  to obtain
       per-channel scale γ ∈ ℝ^{B×C} and shift β ∈ ℝ^{B×C}.
    2. Applies GroupNorm to **h**.
    3. Modulates:  ``γ · GroupNorm(h) + β``   (broadcast over H, W).

    This is the standard conditioning mechanism used in modern diffusion
    U-Nets (DiT, ADM, etc.).

    Args:
        n_channels: Number of spatial feature channels *C*.
        time_embed_dim: Dimensionality of the time-embedding vector *E*.
        n_groups: Number of groups for ``GroupNorm``.  Defaults to 32,
                  which is standard for channel counts that are multiples
                  of 32.
    """

    def __init__(
        self,
        n_channels: int,
        time_embed_dim: int = 256,
        n_groups: int = 32,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels

        # GroupNorm (affine=False because AdaLN provides its own γ/β)
        # Clamp num_groups so it always divides n_channels.
        safe_groups = min(n_groups, n_channels)
        while n_channels % safe_groups != 0:
            safe_groups -= 1
        self.norm = nn.GroupNorm(num_groups=safe_groups, num_channels=n_channels, affine=False)

        # Projection from time embedding → (γ, β)
        # Output has 2·C channels: first C are γ, last C are β.
        self.projection = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * n_channels),
        )

        # Initialise the final linear layer to output zeros so that at
        # init-time γ = 1 and β = 0  (identity modulation).
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    # ------------------------------------------------------------------ #

    def forward(
        self,
        h: torch.Tensor,
        time_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive normalisation.

        Args:
            h: Spatial feature map of shape ``(B, C, H, W)``.
            time_embed: Time embedding of shape ``(B, time_embed_dim)``.

        Returns:
            Modulated feature map of shape ``(B, C, H, W)``.
        """
        # Project embedding → scale & shift
        params = self.projection(time_embed)                       # (B, 2C)
        gamma, beta = params.chunk(2, dim=1)                       # each (B, C)

        # Reshape for spatial broadcasting: (B, C) → (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)                  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)                    # (B, C, 1, 1)

        # GroupNorm → modulate
        # At init, gamma ≈ 0 and beta ≈ 0, so output ≈ norm(h)
        # because we add 1 to gamma to make the initial modulation
        # an identity: (1 + 0) * norm(h) + 0 = norm(h).
        h_norm = self.norm(h)                                      # (B, C, H, W)
        return (1.0 + gamma) * h_norm + beta                       # (B, C, H, W)
