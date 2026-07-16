from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .checkpoint import load_strict_generator
from .config import load_config
from .data import load_cached_sample
from .flow import flow_matching_losses, sample_flow_training_batch
from .model import InterleavedWanAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Full_Duplex_Fix/configs/overfit.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--output", default="Full_Duplex_Fix/outputs/smallest_000000/smoke_model.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
    sample = load_cached_sample(config["cache_path"])
    generator, checkpoint_audit = load_strict_generator(
        config, device=device, dtype=torch.float32
    )
    model = InterleavedWanAdapter(
        generator, gradient_checkpointing=bool(config["gradient_checkpointing"])
    ).train(args.backward)
    tensors = sample.batched(device, dtype)
    rng = torch.Generator(device=device).manual_seed(int(config["fixed_evaluation_seed"]))
    batch = sample_flow_training_batch(tensors["world_latents"], model.scheduler, generator=rng)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    context = torch.enable_grad() if args.backward else torch.inference_mode()
    with context, torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16
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
    forward_seconds = time.perf_counter() - started
    gradient_samples = {}
    backward_seconds = None
    if args.backward:
        started = time.perf_counter()
        losses.total.backward()
        torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - started
        parameters = {
            "patch_embedding": model.backbone.patch_embedding.weight,
            "block_0_q": model.backbone.blocks[0].self_attn.q.weight,
            "block_15_q": model.backbone.blocks[15].self_attn.q.weight,
            "block_29_q": model.backbone.blocks[29].self_attn.q.weight,
            "flow_head": model.backbone.head.head.weight,
        }
        for name, parameter in parameters.items():
            gradient_samples[name] = (
                None if parameter.grad is None else float(parameter.grad.float().norm())
            )
            if gradient_samples[name] is None or not math.isfinite(gradient_samples[name]):
                raise RuntimeError(f"Missing/non-finite gradient for {name}")
    torch.cuda.synchronize(device)
    result = {
        "checkpoint": checkpoint_audit,
        "input_layout": model.layout.name,
        "sequence_length": model.layout.sequence_length,
        "num_blocks": len(model.backbone.blocks),
        "flow_shape": list(output.flow.shape),
        "flow_finite": bool(torch.isfinite(output.flow).all()),
        "loss": float(losses.total.detach()),
        "loss_init": float(losses.init.detach()),
        "loss_transition": float(losses.transition.detach()),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "gradient_samples": gradient_samples,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
