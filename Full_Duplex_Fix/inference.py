from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .checkpoint import load_strict_generator, normalize_generator_state_dict
from .config import load_config
from .data import CachedSample, load_cached_sample
from .training import latent_metrics


@dataclass
class BranchCaches:
    normal: list[dict[str, torch.Tensor]]
    prope: list[dict[str, torch.Tensor]]
    cross: list[dict[str, Any]]


def load_experiment_weights(generator: torch.nn.Module, path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    generator_state = normalize_generator_state_dict(state)
    incompatible = generator.load_state_dict(generator_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected experiment checkpoint mismatch: {incompatible}")
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "global_step": state.get("global_step"),
        "checkpoint_version": state.get("checkpoint_version", "base_generator_checkpoint"),
        "strict_load": True,
    }


class AutoregressiveSampler:
    def __init__(
        self,
        generator: torch.nn.Module,
        config: dict[str, Any],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        self.generator = generator.eval()
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.frame_tokens = int(config["tokens_per_span"])
        self.num_layers = int(config["num_transformer_blocks"])
        self.num_heads = int(config["num_heads"])
        self.head_dim = int(config["head_dim"])
        self.cache_tokens = int(config["local_attention_states"]) * self.frame_tokens
        if self.cache_tokens < int(config["num_states"]) * self.frame_tokens:
            raise ValueError("KV cache must hold all 20 clean states")

    def _branch_caches(self, batch: int) -> BranchCaches:
        def self_entry() -> dict[str, torch.Tensor]:
            return {
                "k": torch.zeros(
                    batch,
                    self.cache_tokens,
                    self.num_heads,
                    self.head_dim,
                    device=self.device,
                    dtype=self.dtype,
                ),
                "v": torch.zeros(
                    batch,
                    self.cache_tokens,
                    self.num_heads,
                    self.head_dim,
                    device=self.device,
                    dtype=self.dtype,
                ),
                "global_end_index": torch.zeros(1, dtype=torch.long, device=self.device),
                "local_end_index": torch.zeros(1, dtype=torch.long, device=self.device),
            }

        normal = [self_entry() for _ in range(self.num_layers)]
        prope = [self_entry() for _ in range(self.num_layers)]
        cross = [
            {
                "k": torch.zeros(
                    batch,
                    int(self.config["text_length"]),
                    self.num_heads,
                    self.head_dim,
                    device=self.device,
                    dtype=self.dtype,
                ),
                "v": torch.zeros(
                    batch,
                    int(self.config["text_length"]),
                    self.num_heads,
                    self.head_dim,
                    device=self.device,
                    dtype=self.dtype,
                ),
                "is_init": False,
            }
            for _ in range(self.num_layers)
        ]
        return BranchCaches(normal=normal, prope=prope, cross=cross)

    def _scheduler(self):
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=int(self.config["num_train_timesteps"]),
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(
            int(self.config["sampling_steps"]),
            device=self.device,
            shift=float(self.config["timestep_shift"]),
        )
        return scheduler

    def _call_branch(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        prompt: torch.Tensor,
        caches: BranchCaches,
        physical_time: int,
        viewmat: torch.Tensor,
        K: torch.Tensor,
    ) -> torch.Tensor:
        flow, _ = self.generator(
            noisy_image_or_video=latent,
            conditional_dict={"prompt_embeds": prompt},
            timestep=timestep,
            kv_cache=caches.normal,
            crossattn_cache=caches.cross,
            current_start=physical_time * self.frame_tokens,
            cache_start=physical_time * self.frame_tokens,
            viewmats=viewmat,
            Ks=K,
            prope_kv_cache=caches.prope,
        )
        return flow

    def _assert_cache_endpoint(
        self,
        caches: BranchCaches,
        expected: int,
        *,
        branch: str,
        phase: str,
    ) -> None:
        for layer, (normal, prope) in enumerate(zip(caches.normal, caches.prope)):
            values = {
                "normal_global": int(normal["global_end_index"].item()),
                "normal_local": int(normal["local_end_index"].item()),
                "prope_global": int(prope["global_end_index"].item()),
                "prope_local": int(prope["local_end_index"].item()),
            }
            if any(value != expected for value in values.values()):
                raise RuntimeError(
                    f"{branch} cache mismatch during {phase}, layer {layer}: "
                    f"{values}, expected={expected}"
                )

    @torch.inference_mode()
    def sample(
        self,
        *,
        initial_noises: torch.Tensor,
        positive_prompt: torch.Tensor,
        negative_prompt: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        show_progress: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch = initial_noises.shape[0]
        expected = (batch, 20, 16, 60, 104)
        if tuple(initial_noises.shape) != expected:
            raise ValueError(f"Expected noise {expected}, got {tuple(initial_noises.shape)}")
        if positive_prompt.shape != (batch, 512, 4096):
            raise ValueError("positive_prompt must be [B,512,4096]")
        if negative_prompt.shape != (batch, 512, 4096):
            raise ValueError("negative_prompt must be [B,512,4096]")
        if viewmats.shape != (batch, 20, 4, 4) or Ks.shape != (batch, 20, 3, 3):
            raise ValueError("Camera tensors must be [B,20,4,4] and [B,20,3,3]")
        positive_caches = self._branch_caches(batch)
        negative_caches = self._branch_caches(batch)
        outputs = []
        state_timings = []
        guidance = float(self.config["guidance_scale"])

        for physical_time in range(20):
            started = time.perf_counter()
            previous_endpoint = physical_time * self.frame_tokens
            self._assert_cache_endpoint(
                positive_caches,
                previous_endpoint,
                branch="positive",
                phase=f"state {physical_time} entry",
            )
            self._assert_cache_endpoint(
                negative_caches,
                previous_endpoint,
                branch="negative",
                phase=f"state {physical_time} entry",
            )
            latent = initial_noises[:, physical_time : physical_time + 1].clone()
            scheduler = self._scheduler()
            iterator = scheduler.timesteps
            if show_progress:
                iterator = tqdm(iterator, desc=f"state {physical_time:02d}", leave=False)
            viewmat = viewmats[:, physical_time : physical_time + 1]
            K = Ks[:, physical_time : physical_time + 1]
            for solver_timestep in iterator:
                timestep = solver_timestep.expand(batch, 1).to(
                    device=self.device, dtype=torch.float32
                )
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=self.dtype == torch.bfloat16,
                ):
                    conditional_flow = self._call_branch(
                        latent,
                        timestep,
                        positive_prompt,
                        positive_caches,
                        physical_time,
                        viewmat,
                        K,
                    )
                    unconditional_flow = self._call_branch(
                        latent,
                        timestep,
                        negative_prompt,
                        negative_caches,
                        physical_time,
                        viewmat,
                        K,
                    )
                    flow = unconditional_flow + guidance * (
                        conditional_flow - unconditional_flow
                    )
                denoising_endpoint = (physical_time + 1) * self.frame_tokens
                self._assert_cache_endpoint(
                    positive_caches,
                    denoising_endpoint,
                    branch="positive",
                    phase=f"state {physical_time} denoising",
                )
                self._assert_cache_endpoint(
                    negative_caches,
                    denoising_endpoint,
                    branch="negative",
                    phase=f"state {physical_time} denoising",
                )
                latent = scheduler.step(
                    flow, solver_timestep, latent, return_dict=False
                )[0]

            outputs.append(latent)
            zero_timestep = torch.zeros(batch, 1, device=self.device, dtype=torch.float32)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.dtype == torch.bfloat16,
            ):
                self._call_branch(
                    latent,
                    zero_timestep,
                    positive_prompt,
                    positive_caches,
                    physical_time,
                    viewmat,
                    K,
                )
                self._call_branch(
                    latent,
                    zero_timestep,
                    negative_prompt,
                    negative_caches,
                    physical_time,
                    viewmat,
                    K,
                )
            expected_endpoint = (physical_time + 1) * self.frame_tokens
            self._assert_cache_endpoint(
                positive_caches,
                expected_endpoint,
                branch="positive",
                phase=f"state {physical_time} clean rerun",
            )
            self._assert_cache_endpoint(
                negative_caches,
                expected_endpoint,
                branch="negative",
                phase=f"state {physical_time} clean rerun",
            )
            state_timings.append(time.perf_counter() - started)

        generated = torch.cat(outputs, dim=1)
        if generated.shape != expected or not torch.isfinite(generated).all():
            raise RuntimeError(f"Invalid generated latent tensor: {generated.shape}")
        provenance = {
            "num_states": 20,
            "sampling_steps": int(self.config["sampling_steps"]),
            "guidance_scale": guidance,
            "solver": "FlowUniPCMultistepScheduler",
            "state_seconds": state_timings,
            "total_seconds": sum(state_timings),
            "ground_truth_latents_used": False,
            "normal_and_prope_cache_end": 20 * self.frame_tokens,
        }
        return generated, provenance


def run_inference(
    config: dict[str, Any],
    *,
    checkpoint: str | Path,
    device: torch.device | str,
    output: str | Path,
    seed: int | None = None,
    show_progress: bool = True,
) -> tuple[Path, dict[str, Any]]:
    device = torch.device(device)
    dtype = torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
    sample: CachedSample = load_cached_sample(config["cache_path"])
    generator, base_audit = load_strict_generator(config, device=device, dtype=dtype)
    experiment_audit = load_experiment_weights(generator, checkpoint)
    tensors = sample.batched(device, dtype)
    evaluation_seed = int(seed if seed is not None else config["fixed_evaluation_seed"])
    noise_generator = torch.Generator(device=device).manual_seed(evaluation_seed)
    initial_noises = torch.randn(
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
    generated, provenance = sampler.sample(
        initial_noises=initial_noises,
        positive_prompt=tensors["prompt_embedding"],
        negative_prompt=tensors["negative_prompt_embedding"],
        viewmats=tensors["viewmats"],
        Ks=tensors["Ks"],
        show_progress=show_progress,
    )
    metrics = latent_metrics(generated, tensors["world_latents"])
    provenance.update(
        {
            "evaluation_seed": evaluation_seed,
            "base_checkpoint": base_audit,
            "experiment_checkpoint": experiment_audit,
            "cache_preprocessing_hash": sample.metadata["preprocessing_hash"],
            "cache_tensor_sha256": sample.metadata["tensor_sha256"],
            "metrics_against_training_sample": metrics,
        }
    )
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "states": generated.cpu(),
            "target_states": sample.world_latents.unsqueeze(0),
            "initial_noise": initial_noises.cpu(),
        },
        output_path,
    )
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Full_Duplex_Fix/configs/overfit.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", default="Full_Duplex_Fix/outputs/smallest_000000/generated_latents.pt"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=None,
        help="Override sampling_steps for a diagnostic run; omit for the configured 50-step run.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.sampling_steps is not None:
        if args.sampling_steps <= 0:
            parser.error("--sampling-steps must be positive")
        config = dict(config)
        config["sampling_steps"] = args.sampling_steps
    output, provenance = run_inference(
        config,
        checkpoint=args.checkpoint,
        device=args.device,
        output=args.output,
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    print(json.dumps({"output": str(output), **provenance}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
