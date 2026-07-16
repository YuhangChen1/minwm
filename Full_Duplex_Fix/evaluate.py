from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_strict_generator
from .config import load_config
from .data import load_cached_sample
from .flow import flow_matching_losses, flow_to_clean, sample_flow_training_batch
from .inference import load_experiment_weights, run_inference
from .model import InterleavedWanAdapter
from .training import latent_metrics


@torch.inference_mode()
def teacher_forced_evaluation(
    config,
    *,
    checkpoint: str | Path,
    device: torch.device,
) -> dict:
    compute_dtype = (
        torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
    )
    sample = load_cached_sample(config["cache_path"])
    generator, base_audit = load_strict_generator(
        config,
        device=device,
        dtype=torch.float32,
    )
    experiment_audit = load_experiment_weights(generator, checkpoint)
    model = InterleavedWanAdapter(generator, gradient_checkpointing=False).eval()
    tensors = sample.batched(device, compute_dtype)
    rng = torch.Generator(device=device).manual_seed(int(config["fixed_evaluation_seed"]))
    batch = sample_flow_training_batch(tensors["world_latents"], model.scheduler, generator=rng)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=compute_dtype == torch.bfloat16,
    ):
        output = model(
            noisy_states=batch.noisy_latents,
            clean_states=tensors["world_latents"],
            noisy_timesteps=batch.timesteps,
            prompt_embedding=tensors["prompt_embedding"],
            viewmats=tensors["viewmats"],
            Ks=tensors["Ks"],
        )
        losses = flow_matching_losses(output.flow, batch.targets, batch.weights)
    clean = flow_to_clean(batch.noisy_latents, output.flow, batch.timesteps, model.scheduler)
    metrics = latent_metrics(clean, tensors["world_latents"])
    metrics.update(
        {
            "evaluation_kind": "teacher_forced_fixed_noise_single_step",
            "flow_loss": float(losses.total),
            "init_flow_loss": float(losses.init),
            "transition_flow_loss": float(losses.transition),
            "per_state_flow_loss": losses.per_state.cpu().tolist(),
            "base_checkpoint": base_audit,
            "experiment_checkpoint": experiment_audit,
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Full_Duplex_Fix/configs/overfit.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--autonomous", action="store_true")
    parser.add_argument(
        "--output", default="Full_Duplex_Fix/outputs/smallest_000000/fresh_evaluation.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    output_path = Path(args.output).resolve()
    if args.autonomous:
        latent_path = output_path.with_suffix(".pt")
        _, metrics = run_inference(
            config,
            checkpoint=args.checkpoint,
            device=args.device,
            output=latent_path,
        )
    else:
        metrics = teacher_forced_evaluation(
            config, checkpoint=args.checkpoint, device=torch.device(args.device)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
