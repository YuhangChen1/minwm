from __future__ import annotations

from dataclasses import dataclass

import torch

from .debug.tracer import debug_timer, trace_event


@dataclass(frozen=True)
class FlowTrainingBatch:
    noise: torch.Tensor
    timestep_indices: torch.Tensor
    timesteps: torch.Tensor
    noisy_latents: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor


@dataclass(frozen=True)
class FlowLosses:
    total: torch.Tensor
    init: torch.Tensor
    transition: torch.Tensor
    per_state: torch.Tensor


def sample_flow_training_batch(
    clean_latents: torch.Tensor,
    scheduler,
    *,
    generator: torch.Generator | None = None,
) -> FlowTrainingBatch:
    if clean_latents.ndim != 5:
        raise ValueError(f"Expected [B,F,C,H,W], got {tuple(clean_latents.shape)}")
    batch, frames = clean_latents.shape[:2]
    trace_event(
        "data",
        "flow.clean_latents",
        tensors={"clean_latents": clean_latents},
        details={"operation": "cached VAE latents enter flow corruption"},
    )
    with debug_timer() as timing:
        timestep_indices = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (batch, frames),
            device=clean_latents.device,
            generator=generator,
        )
        timesteps = scheduler.timesteps.to(clean_latents.device)[timestep_indices]
    trace_event(
        "flow",
        "flow.sample_timesteps",
        tensors={"indices": timestep_indices, "timesteps": timesteps},
        details=timing,
    )
    with debug_timer() as timing:
        noise = torch.randn(
            clean_latents.shape,
            device=clean_latents.device,
            dtype=clean_latents.dtype,
            generator=generator,
        )
    trace_event("flow", "flow.sample_noise", tensors={"noise": noise}, details=timing)
    with debug_timer() as timing:
        noisy = scheduler.add_noise(
            clean_latents.flatten(0, 1),
            noise.flatten(0, 1),
            timesteps.flatten(0, 1),
        ).unflatten(0, (batch, frames))
    trace_event(
        "flow",
        "flow.add_noise",
        tensors={"noisy_latents": noisy},
        details={
            **timing,
            "formula": "N=(1-sigma)*W+sigma*epsilon",
            "shape_change": "[B*20,C,H,W] -> [B,20,C,H,W]",
        },
    )
    with debug_timer() as timing:
        targets = scheduler.training_target(clean_latents, noise, timesteps)
    trace_event(
        "flow",
        "flow.training_target",
        tensors={"target": targets},
        details={**timing, "formula": "epsilon-W"},
    )
    with debug_timer() as timing:
        weights = scheduler.training_weight(timesteps).unflatten(0, (batch, frames))
    trace_event("flow", "flow.training_weight", tensors={"weights": weights}, details=timing)
    return FlowTrainingBatch(
        noise=noise,
        timestep_indices=timestep_indices,
        timesteps=timesteps,
        noisy_latents=noisy,
        targets=targets,
        weights=weights,
    )


def flow_matching_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> FlowLosses:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target mismatch: {prediction.shape} vs {target.shape}")
    if prediction.ndim != 5 or weights.shape != prediction.shape[:2]:
        raise ValueError("Expected prediction [B,F,C,H,W] and weights [B,F]")
    with debug_timer() as timing:
        squared = (prediction.float() - target.float()).square()
        per_example_state = squared.mean(dim=(2, 3, 4)) * weights.float()
        per_state = per_example_state.mean(dim=0)
        total = per_state.mean()
        init = per_state[0]
        transition = per_state[1:].mean()
    trace_event(
        "loss",
        "loss.weighted_flow_mse",
        tensors={
            "prediction": prediction,
            "target": target,
            "weights": weights,
            "per_state": per_state,
            "total": total,
            "init": init,
            "transition": transition,
        },
        details={
            **timing,
            "formula": "mean_state(mean_CHW((prediction-target)^2)*weight)",
        },
    )
    return FlowLosses(total=total, init=init, transition=transition, per_state=per_state)


def scheduler_sigmas(scheduler, timesteps: torch.Tensor) -> torch.Tensor:
    scheduler_timesteps = scheduler.timesteps.to(timesteps.device)
    scheduler_sigmas_tensor = scheduler.sigmas.to(timesteps.device)
    flat = timesteps.reshape(-1)
    indices = torch.argmin(
        (scheduler_timesteps.unsqueeze(0) - flat.unsqueeze(1)).abs(), dim=1
    )
    return scheduler_sigmas_tensor[indices].reshape(timesteps.shape)


def flow_to_clean(
    noisy_latents: torch.Tensor,
    flow_prediction: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
) -> torch.Tensor:
    if noisy_latents.shape != flow_prediction.shape:
        raise ValueError("noisy_latents and flow_prediction must have the same shape")
    sigmas = scheduler_sigmas(scheduler, timesteps)
    while sigmas.ndim < noisy_latents.ndim:
        sigmas = sigmas.unsqueeze(-1)
    return noisy_latents - sigmas.to(noisy_latents.dtype) * flow_prediction
