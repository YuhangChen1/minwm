from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_strict_generator
from .config import load_config
from .data import load_cached_sample
from .flow import sample_flow_training_batch
from .model import InterleavedWanAdapter


def _per_state_difference(left: torch.Tensor, right: torch.Tensor) -> list[float]:
    difference = (left.float() - right.float()).abs()
    return difference.amax(dim=(0, 2, 3, 4)).cpu().tolist()


@torch.inference_mode()
def run(config: dict, *, device: torch.device, physical_time: int) -> dict:
    dtype = torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
    sample = load_cached_sample(config["cache_path"])
    generator, checkpoint_audit = load_strict_generator(config, device=device, dtype=dtype)
    model = InterleavedWanAdapter(generator, gradient_checkpointing=False).eval()
    tensors = sample.batched(device, dtype)
    rng = torch.Generator(device=device).manual_seed(int(config["fixed_evaluation_seed"]))
    flow_batch = sample_flow_training_batch(
        tensors["world_latents"], model.scheduler, generator=rng
    )

    def forward(noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=dtype == torch.bfloat16,
        ):
            return model(
                noisy_states=noisy,
                clean_states=clean,
                noisy_timesteps=flow_batch.timesteps,
                prompt_embedding=tensors["prompt_embedding"],
                viewmats=tensors["viewmats"],
                Ks=tensors["Ks"],
            ).flow

    clean = tensors["world_latents"]
    baseline = forward(flow_batch.noisy_latents, clean)

    modified_clean = clean.clone()
    modified_clean[:, physical_time].add_(0.5)
    clean_difference = _per_state_difference(
        baseline,
        forward(flow_batch.noisy_latents, modified_clean),
    )

    modified_noisy = flow_batch.noisy_latents.clone()
    modified_noisy[:, physical_time].add_(0.5)
    noisy_difference = _per_state_difference(
        baseline,
        forward(modified_noisy, clean),
    )

    tolerance = 1e-6
    if clean_difference[physical_time] > tolerance:
        raise AssertionError(
            f"N{physical_time} leaked from W{physical_time}: "
            f"max error {clean_difference[physical_time]}"
        )
    if clean_difference[physical_time + 1] <= tolerance:
        raise AssertionError(
            f"W{physical_time} did not affect the allowed N{physical_time + 1} path"
        )
    other_noisy = [
        value for index, value in enumerate(noisy_difference) if index != physical_time
    ]
    if max(other_noisy) > tolerance:
        raise AssertionError(
            f"N{physical_time} leaked into another noisy query: max error {max(other_noisy)}"
        )
    if noisy_difference[physical_time] <= tolerance:
        raise AssertionError(f"Perturbing N{physical_time} did not change its own output")

    return {
        "test": "real 40-span no-leakage perturbation",
        "dtype": str(dtype),
        "physical_time": physical_time,
        "tolerance": tolerance,
        "clean_Wt_perturbation_per_noisy_state_max_abs": clean_difference,
        "noisy_Nt_perturbation_per_noisy_state_max_abs": noisy_difference,
        "N_t_unchanged_when_W_t_changes": True,
        "N_t_plus_1_allowed_path_changes": True,
        "other_noisy_states_unchanged_when_N_t_changes": True,
        "checkpoint": checkpoint_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Full_Duplex_Fix/configs/overfit.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-time", type=int, default=5, choices=range(0, 19))
    parser.add_argument(
        "--output",
        default="Full_Duplex_Fix/outputs/smallest_000000/no_leakage.json",
    )
    args = parser.parse_args()
    result = run(
        load_config(args.config),
        device=torch.device(args.device),
        physical_time=args.physical_time,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
