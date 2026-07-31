"""
diffusion_edm — Continuous-Time Score-Based SDE Diffusion Model (EDM Framework).

A complete implementation of the Karras EDM preconditioning framework for
variational diffusion models. All time is treated as a continuous floating-point
scalar t ∈ (0, T]. No discrete Markov chains or DDPM noise schedules.

Modules:
    config          — Shared mathematical definitions and hyperparameters
    time_embedding  — Fourier feature embeddings and AdaLN conditioning
    backbone        — U-Net neural backbone with AdaLN time injection
    model_wrapper   — EDM preconditioning wrapper (algebraic scaling layer)
    train           — Monte Carlo integration training loop
    sample          — Deterministic Probability Flow ODE sampler
"""
