from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_strict_generator
from .config import load_config
from .data import load_cached_sample
from .dataset_cache import PreencodedVideoDataset
from .inference import AutoregressiveSampler, load_experiment_weights
from .training import latent_metrics


def run_dataset_sample_inference(
    config: dict[str, Any],
    *,
    checkpoint: str | Path,
    sample_index: int,
    negative_cache: str | Path,
    device: torch.device | str,
    output: str | Path,
    seed: int,
    show_progress: bool = True,
) -> tuple[Path, dict[str, Any]]:
    device = torch.device(device)
    dtype = torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
    dataset = PreencodedVideoDataset(
        config["dataset_cache_path"],
        indices=[sample_index],
        expected_count=int(config["expected_dataset_size"]),
    )
    sample = dataset[0]
    entry = dataset.entries[0]

    negative_sample = load_cached_sample(negative_cache)
    cached_negative_prompt = negative_sample.metadata.get("negative_prompt")
    if cached_negative_prompt != config["negative_prompt"]:
        raise ValueError("Shared negative-prompt cache does not match this config")

    generator, base_audit = load_strict_generator(config, device=device, dtype=dtype)
    experiment_audit = load_experiment_weights(generator, checkpoint)
    positive_prompt = sample["prompt_embedding"].unsqueeze(0).to(
        device=device, dtype=dtype
    )
    negative_prompt = negative_sample.negative_prompt_embedding.unsqueeze(0).to(
        device=device, dtype=dtype
    )
    viewmats = sample["viewmats"].unsqueeze(0).to(device=device, dtype=dtype)
    Ks = sample["Ks"].unsqueeze(0).to(device=device, dtype=dtype)
    target = sample["world_latents"].unsqueeze(0).to(device=device, dtype=dtype)

    noise_generator = torch.Generator(device=device).manual_seed(seed)
    initial_noise = torch.randn(
        1,
        20,
        16,
        60,
        104,
        device=device,
        dtype=dtype,
        generator=noise_generator,
    )
    sampler = AutoregressiveSampler(generator, config, device=device, dtype=dtype)
    generated, generation = sampler.sample(
        initial_noises=initial_noise,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        viewmats=viewmats,
        Ks=Ks,
        show_progress=show_progress,
    )
    metrics = latent_metrics(generated, target)
    provenance = {
        **generation,
        "evaluation_seed": seed,
        "sample_index": sample_index,
        "caption": entry["caption"],
        "pose_str": entry["pose_str"],
        "source_video": entry["video_path"],
        "ground_truth_latents_used": False,
        "base_checkpoint": base_audit,
        "experiment_checkpoint": experiment_audit,
        "negative_prompt_cache": str(Path(negative_cache).resolve()),
        "metrics_against_training_sample": metrics,
    }
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "states": generated.cpu(),
            "target_states": target.cpu(),
            "initial_noise": initial_noise.cpu(),
        },
        output_path,
    )
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Full_Duplex_Fix/configs/train_50.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-index", required=True, type=int)
    parser.add_argument(
        "--negative-cache", default="Full_Duplex_Fix/cache/smallest_000000"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output, provenance = run_dataset_sample_inference(
        config,
        checkpoint=args.checkpoint,
        sample_index=args.sample_index,
        negative_cache=args.negative_cache,
        device=args.device,
        output=args.output,
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    print(json.dumps({"output": str(output), **provenance}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
