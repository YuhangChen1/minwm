from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_strict_generator
from .config import load_config
from .data import load_cached_sample
from .flow import flow_matching_losses, sample_flow_training_batch
from .model import InterleavedWanAdapter


def run(config: dict, *, device: torch.device) -> dict:
    dtype = torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
    sample = load_cached_sample(config["cache_path"])
    generator, checkpoint_audit = load_strict_generator(config, device=device, dtype=torch.float32)
    model = InterleavedWanAdapter(generator, gradient_checkpointing=True).train()
    tensors = sample.batched(device, dtype)
    rng = torch.Generator(device=device).manual_seed(int(config["fixed_evaluation_seed"]))
    batch = sample_flow_training_batch(tensors["world_latents"], model.scheduler, generator=rng)
    selected = {
        "patch_embedding": model.backbone.patch_embedding.weight,
        "block_0_q": model.backbone.blocks[0].self_attn.q.weight,
        "block_15_q": model.backbone.blocks[15].self_attn.q.weight,
        "block_29_q": model.backbone.blocks[29].self_attn.q.weight,
        "flow_head": model.backbone.head.head.weight,
    }

    def component_gradients(component: str) -> dict[str, float]:
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=dtype == torch.bfloat16,
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
        loss = losses.init if component == "init" else losses.transition
        loss.backward()
        norms = {}
        for name, parameter in selected.items():
            if parameter.grad is None:
                raise AssertionError(f"{component} loss produced no gradient for {name}")
            norm = float(parameter.grad.float().norm())
            if not torch.isfinite(parameter.grad).all() or norm <= 0:
                raise AssertionError(f"Invalid {component} gradient for {name}: {norm}")
            norms[name] = norm
        return norms

    torch.cuda.reset_peak_memory_stats(device)
    init = component_gradients("init")
    transition = component_gradients("transition")
    return {
        "test": "separate L_init and L_transition real gradient audit",
        "dtype": str(dtype),
        "init_gradient_norms": init,
        "transition_gradient_norms": transition,
        "all_selected_gradients_finite_nonzero": True,
        "vae_in_training_graph": False,
        "umt5_in_training_graph": False,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "checkpoint": checkpoint_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Full_Duplex_Fix/configs/overfit.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        default="Full_Duplex_Fix/outputs/smallest_000000/gradient_audit.json",
    )
    args = parser.parse_args()
    result = run(load_config(args.config), device=torch.device(args.device))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
