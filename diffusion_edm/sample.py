"""Deterministic Probability Flow ODE sampler for Karras‑EDM.

Implements Euler and Heun (2nd‑order) solvers that integrate the ODE

    dx/dσ = (x − D_θ(x, σ)) / σ

from  σ_max  down to 0, following the Karras noise schedule.

torch.randn is used **only once**, inside :meth:`ODESampler.sample`, to
draw the initial Gaussian canvas at  σ = σ_max.  No stochastic noise is
injected during integration.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

try:
    from .model_wrapper import EDMPrecond
    from .config import EDMConfig
except ImportError:
    from model_wrapper import EDMPrecond
    from config import EDMConfig


class ODESampler:
    """Deterministic ODE sampler with Karras sigma schedule.

    Parameters
    ----------
    model : EDMPrecond
        Preconditioned denoiser (wrapper around the UNet backbone).
    config : EDMConfig
        EDM hyper‑parameter namespace.  Must expose ``sigma_min``,
        ``sigma_max``, ``rho``, ``img_channels``, and ``img_resolution``.
    """

    def __init__(self, model: EDMPrecond, config: EDMConfig) -> None:
        self.model = model
        self.config = config

    # ------------------------------------------------------------------
    # Sigma schedule
    # ------------------------------------------------------------------

    def get_karras_schedule(self, n_steps: int) -> Tensor:
        r"""Compute the Karras noise schedule.

        .. math::
            \sigma_i = \bigl(
                \sigma_{\min}^{1/\rho}
              + \frac{i}{N-1}
                \bigl(\sigma_{\max}^{1/\rho} - \sigma_{\min}^{1/\rho}\bigr)
            \bigr)^{\rho},
            \qquad i = 0, \dots, N-1

        A final entry  σ_N = 0  is appended so that the ODE is integrated
        all the way to zero noise.

        Parameters
        ----------
        n_steps : int
            Number of discretisation steps *N*.

        Returns
        -------
        Tensor
            Sigma values of shape ``(N + 1,)`` on CPU, dtype float32.
            Ordered from  σ_max  (index 0) down to 0 (index N).
        """
        rho = self.config.rho
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max

        # Indices 0 … N−1.
        steps = torch.arange(n_steps, dtype=torch.float64)
        ramp = steps / (n_steps - 1)  # 0 → 1

        inv_rho_min = sigma_min ** (1.0 / rho)
        inv_rho_max = sigma_max ** (1.0 / rho)

        sigmas = (inv_rho_max + ramp * (inv_rho_min - inv_rho_max)) ** rho
        # Append σ_N = 0.
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        return sigmas.float()

    # ------------------------------------------------------------------
    # ODE right‑hand side
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ode_derivative(self, x: Tensor, sigma: float) -> Tensor:
        """Evaluate the ODE right‑hand side  dx/dσ = (x − D_θ(x, σ)) / σ.

        Parameters
        ----------
        x : Tensor
            Current state ``(B, C, H, W)``.
        sigma : float
            Current scalar noise level (broadcast to batch).

        Returns
        -------
        Tensor
            Derivative ``(B, C, H, W)``, same shape as *x*.
        """
        batch_size = x.shape[0]
        sigma_tensor = x.new_full((batch_size,), sigma)  # (B,)
        denoised = self.model(x, sigma_tensor)
        return (x - denoised) / sigma

    # ------------------------------------------------------------------
    # Euler integrator
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_euler(self, z: Tensor, sigmas: Tensor) -> Tensor:
        """First‑order Euler integration of the probability‑flow ODE.

        Parameters
        ----------
        z : Tensor
            Initial noisy canvas ``(B, C, H, W)`` at  σ = σ_max.
        sigmas : Tensor
            Monotonically decreasing sigma schedule ``(N + 1,)`` ending at 0.

        Returns
        -------
        Tensor
            Denoised samples ``(B, C, H, W)``.
        """
        x = z
        for i in range(len(sigmas) - 1):
            sigma_cur = sigmas[i].item()
            sigma_next = sigmas[i + 1].item()

            d_i = self.ode_derivative(x, sigma_cur)
            x = x + (sigma_next - sigma_cur) * d_i

        return x

    # ------------------------------------------------------------------
    # Heun integrator (2nd order)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_heun(self, z: Tensor, sigmas: Tensor) -> Tensor:
        """Second‑order Heun integration of the probability‑flow ODE.

        For each step *i*:

        1. Euler predictor: ``x̂ = x_i + (σ_{i+1} − σ_i) · d_i``
        2. If σ_{i+1} ≠ 0: trapezoidal corrector with a second derivative
           evaluation at ``(x̂, σ_{i+1})``.
        3. If σ_{i+1} = 0: accept the Euler result (no corrector).

        Parameters
        ----------
        z : Tensor
            Initial noisy canvas ``(B, C, H, W)`` at  σ = σ_max.
        sigmas : Tensor
            Monotonically decreasing sigma schedule ``(N + 1,)`` ending at 0.

        Returns
        -------
        Tensor
            Denoised samples ``(B, C, H, W)``.
        """
        x = z
        for i in range(len(sigmas) - 1):
            sigma_cur = sigmas[i].item()
            sigma_next = sigmas[i + 1].item()
            dt = sigma_next - sigma_cur  # negative (decreasing)

            # 1) Euler predictor.
            d_i = self.ode_derivative(x, sigma_cur)
            x_hat = x + dt * d_i

            # 2) Corrector (trapezoidal) — skip for the final step to 0.
            if sigma_next != 0.0:
                d_hat = self.ode_derivative(x_hat, sigma_next)
                x = x + dt * (d_i + d_hat) / 2.0
            else:
                x = x_hat

        return x

    # ------------------------------------------------------------------
    # High‑level sampling entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        num_steps: int = 20,
        solver: str = "heun",
        seed: int = 42,
    ) -> Tensor:
        """Generate samples from pure noise via the probability‑flow ODE.

        Parameters
        ----------
        num_samples : int
            Number of images to generate.
        num_steps : int
            Number of sigma discretisation steps (``N``).
        solver : str
            Integration method — ``"euler"`` or ``"heun"``.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        Tensor
            Generated images ``(B, C, H, W)`` clipped to ``[0, 1]``.
        """
        # Determine device from model parameters (fall back to CPU).
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        # Sigma schedule.
        sigmas = self.get_karras_schedule(num_steps).to(device)
        sigma_max = sigmas[0].item()

        # --- ONLY place torch.randn is used ---------------------------------
        torch.manual_seed(seed)
        z = torch.randn(
            num_samples,
            self.config.img_channels,
            self.config.img_resolution,
            self.config.img_resolution,
            device=device,
        ) * sigma_max
        # ---------------------------------------------------------------------

        # Integrate ODE.
        if solver == "euler":
            samples = self.sample_euler(z, sigmas)
        elif solver == "heun":
            samples = self.sample_heun(z, sigmas)
        else:
            raise ValueError(
                f"Unknown solver '{solver}'. Choose 'euler' or 'heun'."
            )

        # Training data lives in [-1, 1].  Rescale to [0, 1] for images.
        return (samples * 0.5 + 0.5).clamp(0.0, 1.0)


# --------------------------------------------------------------------------
# Quick smoke test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import dataclasses

    # ------------------------------------------------------------------
    # Minimal stub config — mirrors the real EDMConfig interface.
    # ------------------------------------------------------------------
    @dataclasses.dataclass
    class _StubConfig:
        img_channels: int = 3
        img_resolution: int = 32
        model_channels: int = 32
        channel_mult: tuple = (1, 2)
        sigma_data: float = 0.5
        sigma_min: float = 0.002
        sigma_max: float = 80.0
        rho: float = 7.0
        P_mean: float = -1.2
        P_std: float = 1.2

    # ------------------------------------------------------------------
    # Tiny dummy backbone that simply returns its input (identity).
    # ------------------------------------------------------------------
    class _DummyUNet(nn.Module):
        def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
            return x

    # ------------------------------------------------------------------
    # Minimal EDMPrecond with the dummy backbone swapped in.
    # ------------------------------------------------------------------
    class _DummyPrecond(EDMPrecond):
        def __init__(self, config: _StubConfig) -> None:
            # Skip real UNet construction — inject dummy.
            nn.Module.__init__(self)
            self.sigma_data = config.sigma_data
            self.backbone = _DummyUNet()

    cfg = _StubConfig()
    model = _DummyPrecond(cfg)
    model.eval()

    sampler = ODESampler(model, cfg)

    # --- Determinism check ---
    out1 = sampler.sample(num_samples=2, num_steps=5, solver="heun", seed=42)
    out2 = sampler.sample(num_samples=2, num_steps=5, solver="heun", seed=42)

    l2 = torch.norm(out1 - out2).item()
    print(f"Determinism check — L2 distance between two runs: {l2:.6f}")
    assert l2 == 0.0, f"Non‑deterministic! L2 = {l2}"
    print("✓ Sampler is deterministic.\n")

    # --- Shape check ---
    print(f"Output shape : {out1.shape}")
    print(f"Value range  : [{out1.min().item():.4f}, {out1.max().item():.4f}]")
    print("✓ Smoke test passed.")
