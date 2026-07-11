from __future__ import annotations

import torch


def shifted_sigma(raw_sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """Wan FlowMatchScheduler's monotonic timestep shift."""
    return shift * raw_sigma / (1 + (shift - 1) * raw_sigma)


def denoising_sigmas(
    num_steps: int,
    shift: float,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    raw = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
    sigmas = shifted_sigma(raw, shift)
    sigmas[0] = 1.0
    sigmas[-1] = 0.0
    if not torch.all(sigmas[:-1] > sigmas[1:]):
        raise AssertionError("Denoising sigma schedule must be strictly decreasing")
    return sigmas


def add_flow_noise(clean: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Checkpoint-compatible interpolation: x_sigma=(1-sigma)x0+sigma*epsilon."""
    return (1 - sigma) * clean + sigma * noise


def flow_target(clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Checkpoint-compatible velocity target: epsilon-x0."""
    return noise - clean


def flow_step(
    sample: torch.Tensor,
    predicted_flow: torch.Tensor,
    sigma: torch.Tensor,
    next_sigma: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Euler update matching FlowMatchScheduler.step()."""
    return sample + predicted_flow * (next_sigma - sigma)
