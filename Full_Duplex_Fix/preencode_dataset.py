from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .config import load_config
from .dataset_cache import (
    load_aligned_input_manifest,
    validate_preencoded_tensors,
    write_cache_manifest,
)
from .preencode import LATENT_MEAN, LATENT_STD, _camera_data, _decode_video


class FrozenWanDatasetPreencoder:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        self.dtype = (
            torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
        )
        project_root = Path(config["project_root"])
        for path in (project_root, project_root / "Wan21", project_root / "shared"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from wan.modules.t5 import umt5_xxl
        from wan.modules.tokenizers import HuggingfaceTokenizer
        from wan.modules.vae import _video_vae

        self.vae = _video_vae(pretrained_path=config["vae_checkpoint"], z_dim=16)
        self.vae = self.vae.eval().requires_grad_(False).to(
            device=device, dtype=self.dtype
        )
        self.latent_mean = torch.tensor(
            LATENT_MEAN, device=device, dtype=self.dtype
        )
        self.latent_inverse_std = torch.tensor(
            LATENT_STD, device=device, dtype=self.dtype
        ).reciprocal()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=self.dtype,
            device=device,
        ).eval().requires_grad_(False)
        state = torch.load(
            config["t5_checkpoint"], map_location="cpu", mmap=True, weights_only=True
        )
        incompatible = self.text_encoder.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected strict T5 mismatch: {incompatible}")
        del state
        self.tokenizer = HuggingfaceTokenizer(
            name=config["t5_tokenizer"],
            seq_len=int(config["text_length"]),
            clean="whitespace",
            local_files_only=True,
        )

    @torch.inference_mode()
    def encode_video(self, video_path: str) -> torch.Tensor:
        sample_config = dict(self.config)
        sample_config["video_path"] = video_path
        pixels, _ = _decode_video(sample_config)
        pixels = pixels.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.dtype == torch.bfloat16,
        ):
            latent = self.vae.encode(
                pixels, [self.latent_mean, self.latent_inverse_std]
            ).float()
        latent = latent.permute(0, 2, 1, 3, 4).contiguous()[0].cpu().half()
        if tuple(latent.shape) != (20, 16, 60, 104):
            raise ValueError(f"Unexpected Wan VAE output: {tuple(latent.shape)}")
        return latent

    @torch.inference_mode()
    def encode_caption(self, caption: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids, mask = self.tokenizer(
            [caption], return_mask=True, add_special_tokens=True
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.dtype == torch.bfloat16,
        ):
            context = self.text_encoder(ids, mask)
        length = int(mask[0].gt(0).sum().item())
        context[0, length:] = 0
        return (
            context[0].to(device="cpu", dtype=torch.bfloat16).contiguous(),
            mask[0].bool().cpu().contiguous(),
        )

    def encode_sample(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        world_latents = self.encode_video(sample["video_path"])
        prompt_embedding, prompt_attention_mask = self.encode_caption(sample["caption"])
        viewmats, Ks = _camera_data(sample["pose_str"])
        tensors = {
            "world_latents": world_latents,
            "prompt_embedding": prompt_embedding,
            "prompt_attention_mask": prompt_attention_mask,
            "viewmats": viewmats,
            "Ks": Ks,
        }
        validate_preencoded_tensors(tensors, sample_index=int(sample["index"]))
        return tensors


def _cache_is_valid(path: Path, sample_index: int) -> bool:
    if not path.is_file():
        return False
    try:
        tensors = torch.load(path, map_location="cpu", weights_only=True)
        validate_preencoded_tensors(tensors, sample_index=sample_index)
    except (OSError, RuntimeError, KeyError, TypeError, ValueError, EOFError):
        return False
    return True


def _distributed_context(config: dict[str, Any]) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    expected_world_size = int(config["world_size"])
    if world_size != expected_world_size:
        raise RuntimeError(
            f"preencode_dataset requires torchrun with {expected_world_size} processes; "
            f"got WORLD_SIZE={world_size}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, device


def preencode_dataset(config: dict[str, Any], *, force: bool = False) -> Path | None:
    if config.get("training_mode") != "multi_sample":
        raise ValueError("preencode_dataset requires training_mode=multi_sample")
    rank, _, world_size, device = _distributed_context(config)
    try:
        samples = load_aligned_input_manifest(
            config["input_manifest"],
            project_root=config["project_root"],
            expected_count=int(config["expected_dataset_size"]),
        )
        cache_root = Path(config["dataset_cache_path"])
        sample_root = cache_root / "samples"
        sample_root.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        assigned = samples[rank::world_size]
        pending = [
            sample
            for sample in assigned
            if force
            or not _cache_is_valid(
                sample_root / f"{int(sample['index']):06d}.pt", int(sample["index"])
            )
        ]
        encoder = FrozenWanDatasetPreencoder(config, device) if pending else None
        for local_index, sample in enumerate(pending, start=1):
            started = time.perf_counter()
            index = int(sample["index"])
            output_path = sample_root / f"{index:06d}.pt"
            temporary = output_path.with_suffix(".pt.tmp")
            tensors = encoder.encode_sample(sample)  # type: ignore[union-attr]
            torch.save(tensors, temporary)
            os.replace(temporary, output_path)
            del tensors
            gc.collect()
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "sample_index": index,
                        "rank_progress": f"{local_index}/{len(pending)}",
                        "seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

        manifest_path = None
        if rank == 0:
            manifest_path = write_cache_manifest(
                cache_root,
                input_manifest=config["input_manifest"],
                samples=samples,
            )
            print(json.dumps({"cache_manifest": str(manifest_path)}, sort_keys=True))
        dist.barrier()
        return manifest_path
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="Full_Duplex_Fix/configs/train_1000.yaml"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    preencode_dataset(config, force=args.force)


if __name__ == "__main__":
    main()
