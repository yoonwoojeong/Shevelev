"""
Compact U-Net backbone for 32×32 CIFAR-10 images with AdaLN time conditioning.

This module implements the raw neural network F_θ used inside the Karras EDM
preconditioning wrapper.  The backbone is **strictly bounded**: it contains no
1/σ² scaling or any other algebraic preconditioning — all of that is handled
externally by the wrapper that computes D_θ(x, σ).

Architecture overview
─────────────────────
  Input stem  →  Encoder (3 levels)  →  Bottleneck  →  Decoder (3 levels)  →  Output head

  • Encoder levels operate at 32×32, 16×16, and 8×8.
  • Bottleneck adds self-attention at 8×8.
  • Decoder mirrors the encoder with skip connections (concatenation).
  • All normalization uses GroupNorm; time conditioning uses AdaptiveLayerNorm.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Import FourierFeatures and AdaptiveLayerNorm from the sibling module.
# ---------------------------------------------------------------------------
try:
    from .time_embedding import FourierFeatures, AdaptiveLayerNorm
except ImportError:
    from time_embedding import FourierFeatures, AdaptiveLayerNorm


# ═══════════════════════════════════════════════════════════════════════════
# Helper: safe GroupNorm that caps num_groups at *channels*
# ═══════════════════════════════════════════════════════════════════════════

def _group_norm(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """Return a GroupNorm layer, clamping *num_groups* so it divides *channels*."""
    num_groups = min(num_groups, channels)
    # Ensure num_groups divides channels evenly.
    while channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels)


# ═══════════════════════════════════════════════════════════════════════════
# ResBlock — residual block with AdaLN time conditioning
# ═══════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """Pre-activation residual block with adaptive layer-norm time conditioning.

    Structure::

        AdaLN → SiLU → Conv3×3 → AdaLN → SiLU → Conv3×3  (+)  skip
                                                              ↑
                                                        identity or 1×1

    Parameters
    ----------
    in_channels : int
        Number of input feature-map channels.
    out_channels : int
        Number of output feature-map channels.
    time_embed_dim : int
        Dimensionality of the time-embedding vector fed to AdaLN.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
    ) -> None:
        super().__init__()

        # --- first half ---
        self.adaln1 = AdaptiveLayerNorm(in_channels, time_embed_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # --- second half ---
        self.adaln2 = AdaptiveLayerNorm(out_channels, time_embed_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # --- skip projection (identity when channels match) ---
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Feature map of shape ``(B, C_in, H, W)``.
        time_emb : Tensor
            Time-embedding vector of shape ``(B, time_embed_dim)``.

        Returns
        -------
        Tensor
            Feature map of shape ``(B, C_out, H, W)``.
        """
        h = self.adaln1(x, time_emb)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.adaln2(h, time_emb)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip_proj(x)


# ═══════════════════════════════════════════════════════════════════════════
# SelfAttention — multi-head self-attention over spatial dims
# ═══════════════════════════════════════════════════════════════════════════

class SelfAttention(nn.Module):
    """Multi-head self-attention for 2-D feature maps.

    The spatial dimensions ``(H, W)`` are flattened to a sequence of length
    ``H·W`` before attention and reshaped back afterwards.  A residual
    connection and LayerNorm are applied around the attention.

    Parameters
    ----------
    channels : int
        Number of feature-map channels (used as the embedding dimension).
    num_heads : int
        Number of attention heads.  Must divide *channels*.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Feature map of shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Feature map of same shape, after self-attention + residual.
        """
        B, C, H, W = x.shape

        # (B, C, H, W) → (B, H*W, C)
        h = x.reshape(B, C, H * W).permute(0, 2, 1)

        # LayerNorm → MHA (pre-norm formulation)
        h_norm = self.norm(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, need_weights=False)

        # Residual
        h = h + attn_out

        # (B, H*W, C) → (B, C, H, W)
        return h.permute(0, 2, 1).reshape(B, C, H, W)


# ═══════════════════════════════════════════════════════════════════════════
# Downsample / Upsample helpers
# ═══════════════════════════════════════════════════════════════════════════

class Downsample(nn.Module):
    """Spatial 2× down-sampling via strided convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Downsample ``(B, C, H, W)`` → ``(B, C, H/2, W/2)``."""
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial 2× up-sampling via nearest interpolation + convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Upsample ``(B, C, H, W)`` → ``(B, C, 2H, 2W)``."""
        return self.conv(self.up(x))


# ═══════════════════════════════════════════════════════════════════════════
# UNet — full backbone
# ═══════════════════════════════════════════════════════════════════════════

class UNet(nn.Module):
    """Compact U-Net backbone for Karras EDM on 32×32 CIFAR-10.

    This network is the raw ``F_θ`` in the EDM formulation.  It receives
    *pre-scaled* input ``c_in(σ) · x_t`` and the raw noise level ``σ``,
    and produces a raw prediction tensor.  All algebraic preconditioning
    (``c_skip``, ``c_out``, ``c_in``, ``c_noise``) is applied **externally**
    by the EDM wrapper.

    Parameters
    ----------
    img_channels : int
        Number of image channels (3 for CIFAR-10).
    model_channels : int
        Base channel width of the network (``C``).
    channel_mult : tuple of int
        Per-level channel multipliers.  ``(1, 2, 2)`` gives widths
        ``[C, 2C, 2C]`` at resolutions ``[32, 16, 8]``.
    time_embed_dim : int
        Dimensionality of the time-embedding vector.
    """

    def __init__(
        self,
        img_channels: int = 3,
        model_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2),
        time_embed_dim: int = 256,
    ) -> None:
        super().__init__()

        self.img_channels = img_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.time_embed_dim = time_embed_dim

        num_levels = len(channel_mult)
        ch_list = [model_channels * m for m in channel_mult]  # [128, 256, 256]

        # ── Time embedding ────────────────────────────────────────────────
        self.time_embed = FourierFeatures(time_embed_dim=time_embed_dim)

        # ── Input stem ────────────────────────────────────────────────────
        self.input_conv = nn.Conv2d(
            img_channels, model_channels, kernel_size=3, padding=1,
        )

        # ── Encoder ──────────────────────────────────────────────────────
        self.enc_blocks: nn.ModuleList = nn.ModuleList()
        self.downsamplers: nn.ModuleList = nn.ModuleList()

        in_ch = model_channels
        for level in range(num_levels):
            out_ch = ch_list[level]
            # Two ResBlocks per level
            self.enc_blocks.append(ResBlock(in_ch, out_ch, time_embed_dim))
            self.enc_blocks.append(ResBlock(out_ch, out_ch, time_embed_dim))
            in_ch = out_ch
            # Downsample between levels (not after the last)
            if level < num_levels - 1:
                self.downsamplers.append(Downsample(out_ch))

        # ── Bottleneck ───────────────────────────────────────────────────
        bottleneck_ch = ch_list[-1]
        self.bottleneck_res1 = ResBlock(bottleneck_ch, bottleneck_ch, time_embed_dim)
        self.bottleneck_attn = SelfAttention(bottleneck_ch, num_heads=4)
        self.bottleneck_res2 = ResBlock(bottleneck_ch, bottleneck_ch, time_embed_dim)

        # ── Decoder ──────────────────────────────────────────────────────
        self.dec_blocks: nn.ModuleList = nn.ModuleList()
        self.upsamplers: nn.ModuleList = nn.ModuleList()

        for level in reversed(range(num_levels)):
            out_ch = ch_list[level]
            # Two ResBlocks per level — input channels are doubled by skip concat
            self.dec_blocks.append(ResBlock(in_ch + out_ch, out_ch, time_embed_dim))
            self.dec_blocks.append(ResBlock(out_ch + out_ch, out_ch, time_embed_dim))
            in_ch = out_ch
            # Upsample between levels (not after the first = lowest-res level)
            if level > 0:
                self.upsamplers.append(Upsample(out_ch))

        # ── Output head ──────────────────────────────────────────────────
        self.out_norm = _group_norm(model_channels)
        self.out_conv = nn.Conv2d(
            model_channels, img_channels, kernel_size=3, padding=1,
        )

    # ------------------------------------------------------------------
    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Run the U-Net backbone.

        Parameters
        ----------
        x : Tensor
            Pre-scaled noisy input ``c_in(σ) · x_t`` of shape ``(B, C, H, W)``.
        sigma : Tensor
            Raw noise level of shape ``(B,)``.  The ``FourierFeatures`` module
            internally applies ``c_noise(σ) = ¼ ln(σ)`` before embedding.

        Returns
        -------
        Tensor
            Raw network prediction of shape ``(B, C, H, W)``.  This is
            *not* the final denoised image — the EDM wrapper combines it
            with the skip-connection algebra to form ``D_θ``.
        """
        # Time embedding: (B,) → (B, time_embed_dim)
        t_emb = self.time_embed(sigma)

        # Input stem
        h = self.input_conv(x)

        # ── Encoder ── save skip tensors ──────────────────────────────
        skips: List[Tensor] = []
        enc_idx = 0
        num_levels = len(self.channel_mult)
        ds_idx = 0

        for level in range(num_levels):
            # Two ResBlocks
            h = self.enc_blocks[enc_idx](h, t_emb)
            skips.append(h)
            enc_idx += 1

            h = self.enc_blocks[enc_idx](h, t_emb)
            skips.append(h)
            enc_idx += 1

            # Downsample (except last level)
            if level < num_levels - 1:
                h = self.downsamplers[ds_idx](h)
                ds_idx += 1

        # ── Bottleneck ────────────────────────────────────────────────
        h = self.bottleneck_res1(h, t_emb)
        h = self.bottleneck_attn(h)
        h = self.bottleneck_res2(h, t_emb)

        # ── Decoder ── pop skips in reverse ───────────────────────────
        dec_idx = 0
        us_idx = 0

        for level in reversed(range(num_levels)):
            # First ResBlock: concat with skip
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = self.dec_blocks[dec_idx](h, t_emb)
            dec_idx += 1

            # Second ResBlock: concat with skip
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = self.dec_blocks[dec_idx](h, t_emb)
            dec_idx += 1

            # Upsample (except first = lowest-res level)
            if level > 0:
                h = self.upsamplers[us_idx](h)
                us_idx += 1

        # ── Output head ──────────────────────────────────────────────
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)

        return h
