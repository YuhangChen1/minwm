from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .checkpoint import load_strict_generator
from .config import load_config
from .data import load_cached_sample
from .layout import InterleavedLayout
from .model import InterleavedWanAdapter


def _comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_float = left.float()
    right_float = right.float()
    difference = (left_float - right_float).abs()
    denominator = right_float.abs().mean().clamp_min(1e-12)
    return {
        "max_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "relative_mean_error": float(difference.mean() / denominator),
    }


def _tolerances(dtype: torch.dtype) -> dict[str, float]:
    if dtype == torch.bfloat16:
        return {
            "native_original_max_absolute_error": 0.0,
            "interleaved_max_absolute_error": 0.2,
            "interleaved_relative_mean_error": 0.01,
        }
    return {
        "native_original_max_absolute_error": 2e-3,
        "interleaved_max_absolute_error": 2e-3,
        "interleaved_relative_mean_error": 5e-5,
    }


def assess_result(result: dict, dtype: torch.dtype) -> dict:
    tolerances = _tolerances(dtype)
    original = result["original_vs_custom_original"]
    interleaved = result["custom_original_vs_interleaved"]
    result["declared_tolerances"] = tolerances
    result["passed"] = bool(
        result["finite"]
        and original["max_absolute_error"]
        <= tolerances["native_original_max_absolute_error"]
        and interleaved["max_absolute_error"]
        <= tolerances["interleaved_max_absolute_error"]
        and interleaved["relative_mean_error"]
        <= tolerances["interleaved_relative_mean_error"]
    )
    baseline_mean = original["mean_absolute_error"]
    result["interleaved_to_native_mean_error_ratio"] = (
        None if baseline_mean == 0 else interleaved["mean_absolute_error"] / baseline_mean
    )
    return result


@torch.inference_mode()
def run(config: dict, device: torch.device, dtype: torch.dtype) -> dict:
    sample = load_cached_sample(config["cache_path"])
    generator, checkpoint_audit = load_strict_generator(config, device=device, dtype=dtype)
    generator.eval()
    generator.model.num_frame_per_block = 4
    adapter = InterleavedWanAdapter(generator, gradient_checkpointing=False).eval()
    tensors = sample.batched(device, dtype)
    clean = tensors["world_latents"]
    rng = torch.Generator(device=device).manual_seed(int(config["fixed_evaluation_seed"]))
    noise = torch.randn(clean.shape, device=device, dtype=dtype, generator=rng)
    block_indices = torch.randint(
        0, 1000, (1, 5), device=device, generator=rng
    ).repeat_interleave(4, dim=1)
    timesteps = generator.scheduler.timesteps.to(device)[block_indices]
    noisy = generator.scheduler.add_noise(
        clean.flatten(0, 1), noise.flatten(0, 1), timesteps.flatten(0, 1)
    ).unflatten(0, clean.shape[:2])
    prompt = {"prompt_embeds": tensors["prompt_embedding"]}

    timings = {}
    generator.model.block_mask = None
    started = time.perf_counter()
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16
    ):
        original_flow, _ = generator(
            noisy_image_or_video=noisy,
            conditional_dict=prompt,
            timestep=timesteps,
            clean_x=clean,
            aug_t=None,
            viewmats=tensors["viewmats"],
            Ks=tensors["Ks"],
        )
    torch.cuda.synchronize(device)
    timings["original_seconds"] = time.perf_counter() - started

    layouts = {
        "custom_original": InterleavedLayout.full_teacher_forcing("original"),
        "custom_interleaved": InterleavedLayout.full_teacher_forcing("interleaved"),
    }
    outputs = {}
    for name, layout in layouts.items():
        started = time.perf_counter()
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16
        ):
            outputs[name] = adapter(
                noisy_states=noisy,
                clean_states=clean,
                noisy_timesteps=timesteps,
                prompt_embedding=tensors["prompt_embedding"],
                viewmats=tensors["viewmats"],
                Ks=tensors["Ks"],
                layout=layout,
            ).flow
        torch.cuda.synchronize(device)
        timings[f"{name}_seconds"] = time.perf_counter() - started

    original_comparison = _comparison(original_flow, outputs["custom_original"])
    interleaved_comparison = _comparison(
        outputs["custom_original"], outputs["custom_interleaved"]
    )
    result = {
        "test": "4-state attention-graph permutation equivalence",
        "dtype": str(dtype),
        "checkpoint": checkpoint_audit,
        "timings": timings,
        "original_vs_custom_original": original_comparison,
        "original_vs_custom_interleaved": _comparison(original_flow, outputs["custom_interleaved"]),
        "custom_original_vs_interleaved": interleaved_comparison,
        "output_shape": list(original_flow.shape),
        "finite": bool(
            torch.isfinite(original_flow).all()
            and all(torch.isfinite(output).all() for output in outputs.values())
        ),
    }
    return assess_result(result, dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Full_Duplex_Fix/configs/overfit.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--assess-existing",
        action="store_true",
        help="Apply current declared tolerances to an existing output without rerunning Wan.",
    )
    parser.add_argument(
        "--output",
        default="Full_Duplex_Fix/outputs/smallest_000000/permutation_equivalence.json",
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.assess_existing:
        with output.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        stored_dtype = result.get("dtype")
        if stored_dtype not in {"torch.bfloat16", "torch.float32"}:
            raise ValueError(f"Unsupported stored dtype: {stored_dtype}")
        dtype = torch.bfloat16 if stored_dtype == "torch.bfloat16" else torch.float32
        result = assess_result(result, dtype)
    else:
        config = load_config(args.config)
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        result = run(config, torch.device(args.device), dtype)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit("Permutation-equivalence result exceeded the declared tolerance")


if __name__ == "__main__":
    main()
