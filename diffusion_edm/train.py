"""
train.py — Monte Carlo Integration Training Loop for EDM Diffusion Model.

Implements continuous-time training with log-normal noise sampling,
EDM preconditioning, and thermodynamic loss weighting. No discrete
Markov chains or step-based noise schedules are used anywhere.

Reference: Karras et al., "Elucidating the Design Space of Diffusion-Based
Generative Models" (arXiv:2206.00364), Table 1 & Algorithm 1.
"""

import argparse
import copy
import inspect
import math
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

try:
    from .model_wrapper import EDMPrecond
    from .config import EDMConfig
except ImportError:
    from model_wrapper import EDMPrecond
    from config import EDMConfig


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) Helper
# ---------------------------------------------------------------------------

class EMATracker:
    """Maintains an exponential moving average of model parameters.

    The EMA model is used exclusively for evaluation / sampling and is
    never trained directly.  Decay rate controls how quickly the average
    forgets old weights (higher = slower update).

    Args:
        model: The live training model whose parameters are tracked.
        decay: EMA decay factor in (0, 1). Default 0.9999.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        """Copy shadow parameters into a model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def get_cifar10_loader(
    batch_size: int,
    data_dir: str = "./data",
    num_workers: int = 2,
) -> DataLoader:
    """Create a CIFAR-10 training dataloader with normalization to [-1, 1].

    Args:
        batch_size: Number of images per batch.
        data_dir: Path to download / cache CIFAR-10.
        num_workers: DataLoader worker processes.

    Returns:
        A DataLoader yielding (images, labels) where images ∈ [-1, 1].
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),                        # [0, 1]
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),  # [-1, 1]
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


# ---------------------------------------------------------------------------
# Training Step (Monte Carlo Integration)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sample_sigma(
    batch_size: int,
    P_mean: float,
    P_std: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample noise levels from the log-normal training distribution.

    ln(σ) ~ N(P_mean, P_std²)  ⟹  σ = exp(P_mean + P_std · z),  z ~ N(0, 1)

    This is a *continuous* Monte Carlo draw — no discrete schedule.

    Args:
        batch_size: Number of σ values to sample.
        P_mean: Mean of ln(σ).
        P_std: Std dev of ln(σ).
        device: Target device.

    Returns:
        Tensor of shape (batch_size,) with positive float σ values.
    """
    ln_sigma = P_mean + P_std * torch.randn(batch_size, device=device)
    return ln_sigma.exp()


def training_step(
    model: EDMPrecond,
    images: torch.Tensor,
    config: EDMConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Execute one Monte Carlo training step.

    Procedure (no sequential diffusion chain):
        1. Sample continuous σ from log-normal distribution.
        2. Add Gaussian noise: x_t = x_0 + σ · ε.
        3. Predict denoised image via EDM preconditioning wrapper.
        4. Compute thermodynamically-weighted MSE loss.

    Args:
        model: The EDM-preconditioned model.
        images: Clean training images of shape (B, C, H, W), in [-1, 1].
        config: EDM configuration with noise distribution parameters.

    Returns:
        Tuple of (scalar loss, metrics dict with diagnostics).
    """
    device = images.device
    batch_size = images.shape[0]

    # --- Step 1: Sample continuous noise level σ (Monte Carlo) ---
    sigma = _sample_sigma(batch_size, config.P_mean, config.P_std, device)

    # --- Step 2: Add continuous Gaussian noise ---
    epsilon = torch.randn_like(images)
    # x_t = x_0 + σ · ε  (σ broadcast to spatial dims)
    sigma_spatial = sigma.view(-1, 1, 1, 1)
    x_t = images + sigma_spatial * epsilon

    # --- Step 3: Forward through EDM preconditioning wrapper ---
    # The wrapper internally applies c_skip, c_in, c_out, c_noise
    denoised = model(x_t, sigma)

    # --- Step 4: Compute thermodynamic loss weight ---
    # λ(σ) = (σ² + σ_data²) / (σ² · σ_data²)
    loss_weight = model.get_loss_weight(sigma)  # shape: (B,)

    # --- Step 5: Weighted MSE loss ---
    # Per-sample MSE over spatial dims, then weight by λ(σ)
    mse_per_sample = (denoised - images).pow(2).mean(dim=(1, 2, 3))
    weighted_loss = (loss_weight * mse_per_sample).mean()

    # --- Diagnostics ---
    metrics = {
        "loss": weighted_loss.item(),
        "loss_unweighted": mse_per_sample.mean().item(),
        "sigma_mean": sigma.mean().item(),
        "sigma_std": sigma.std().item(),
        "loss_weight_mean": loss_weight.mean().item(),
        "loss_variance": (loss_weight * mse_per_sample).var().item(),
    }

    return weighted_loss, metrics


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train(config: EDMConfig, args: argparse.Namespace) -> None:
    """Full training loop with EMA tracking and periodic logging.

    Args:
        config: EDM configuration.
        args: Command-line arguments (epochs, device, etc.).
    """
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[train] Using device: {device}")
    print(f"[train] Config: sigma_data={config.sigma_data}, "
          f"P_mean={config.P_mean}, P_std={config.P_std}")

    # --- Performance: cuDNN benchmark for fixed input sizes (32×32) ---
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # --- Model (channels-last for Tensor Core-friendly layout) ---
    model = EDMPrecond(config).to(device, memory_format=torch.channels_last)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {param_count:,}")

    # --- Performance: torch.compile for kernel fusion ---
    if device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("[train] Model compiled with torch.compile")

    # --- EMA ---
    ema = EMATracker(model, decay=config.ema_decay)

    # --- Optimizer (fused CUDA kernel when available) ---
    fused = (
        device.type == "cuda"
        and "fused" in inspect.signature(torch.optim.AdamW).parameters
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        fused=fused,
    )
    if fused:
        print("[train] Using fused AdamW optimizer")

    # --- Data ---
    loader = get_cifar10_loader(
        batch_size=config.batch_size,
        data_dir=args.data_dir,
    )

    # --- LR Schedule (cosine annealing) ---
    total_steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6,
    )

    # --- Performance: AMP (automatic mixed precision) ---
    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    # GradScaler is only needed for FP16 (BF16 doesn't need loss scaling).
    scaler = torch.amp.GradScaler(
        enabled=use_amp and amp_dtype == torch.float16,
    )
    if use_amp:
        print(f"[train] AMP enabled with dtype={amp_dtype}")

    # --- Training ---
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_variance = 0.0

        for batch_idx, (images, _labels) in enumerate(loader):
            images = images.to(device, memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass under AMP autocast
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp,
            ):
                loss, metrics = training_step(model, images, config)

            # Backward pass with loss scaling (no-op for BF16)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            epoch_loss += metrics["loss"]
            epoch_variance += metrics["loss_variance"]
            global_step += 1

            if batch_idx % args.log_every == 0:
                lr_current = scheduler.get_last_lr()[0]
                print(
                    f"  [step {global_step:>6d}] "
                    f"loss={metrics['loss']:.4f}  "
                    f"loss_var={metrics['loss_variance']:.4f}  "
                    f"σ_mean={metrics['sigma_mean']:.3f}  "
                    f"σ_std={metrics['sigma_std']:.3f}  "
                    f"λ_mean={metrics['loss_weight_mean']:.2f}  "
                    f"lr={lr_current:.2e}"
                )

        avg_loss = epoch_loss / len(loader)
        avg_var = epoch_variance / len(loader)
        print(
            f"[epoch {epoch + 1}/{args.epochs}] "
            f"avg_loss={avg_loss:.4f}  avg_loss_variance={avg_var:.4f}"
        )

        # --- Save checkpoint ---
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.ckpt_dir, f"edm_epoch{epoch + 1:04d}.pt")
            os.makedirs(args.ckpt_dir, exist_ok=True)

            # Save EMA weights for sampling
            ema_model = copy.deepcopy(model)
            ema.apply_to(ema_model)

            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, ckpt_path)
            print(f"  [saved] {ckpt_path}")
            del ema_model

    print("[train] Training complete.")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train a Continuous-Time EDM Diffusion Model on CIFAR-10."
    )
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu' (default: cuda)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="CIFAR-10 data directory (default: ./data)")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints",
                        help="Checkpoint save directory (default: ./checkpoints)")
    parser.add_argument("--log_every", type=int, default=50,
                        help="Log metrics every N batches (default: 50)")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs (default: 10)")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    cfg = EDMConfig(
        batch_size=cli_args.batch_size,
        learning_rate=cli_args.lr,
    )
    train(cfg, cli_args)
