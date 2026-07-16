from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class CachedSample:
    world_latents: torch.Tensor
    prompt_embedding: torch.Tensor
    prompt_attention_mask: torch.Tensor
    negative_prompt_embedding: torch.Tensor
    negative_prompt_attention_mask: torch.Tensor
    viewmats: torch.Tensor
    Ks: torch.Tensor
    metadata: dict[str, Any]

    def validate(self) -> None:
        expected = {
            "world_latents": (20, 16, 60, 104),
            "prompt_embedding": (512, 4096),
            "prompt_attention_mask": (512,),
            "negative_prompt_embedding": (512, 4096),
            "negative_prompt_attention_mask": (512,),
            "viewmats": (20, 4, 4),
            "Ks": (20, 3, 3),
        }
        for name, shape in expected.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name}: expected {shape}, got {tuple(tensor.shape)}")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise FloatingPointError(f"{name} contains NaN or Inf")
        if torch.count_nonzero(
            self.prompt_embedding[~self.prompt_attention_mask.bool()]
        ).item():
            raise ValueError("Positive prompt padding embeddings must be zero")
        if torch.count_nonzero(
            self.negative_prompt_embedding[~self.negative_prompt_attention_mask.bool()]
        ).item():
            raise ValueError("Negative prompt padding embeddings must be zero")
        expected_hashes = self.metadata.get("tensor_sha256")
        if not isinstance(expected_hashes, dict):
            raise ValueError("Cache metadata is missing per-tensor SHA256 identities")
        for name in expected:
            actual_hash = tensor_sha256(getattr(self, name))
            if expected_hashes.get(name) != actual_hash:
                raise ValueError(f"Cache tensor SHA256 mismatch for {name}")

    def batched(self, device: torch.device | str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        return {
            "world_latents": self.world_latents.unsqueeze(0).to(device=device, dtype=dtype),
            "prompt_embedding": self.prompt_embedding.unsqueeze(0).to(device=device, dtype=dtype),
            "prompt_attention_mask": self.prompt_attention_mask.unsqueeze(0).to(device=device),
            "negative_prompt_embedding": self.negative_prompt_embedding.unsqueeze(0).to(
                device=device, dtype=dtype
            ),
            "negative_prompt_attention_mask": self.negative_prompt_attention_mask.unsqueeze(0).to(
                device=device
            ),
            "viewmats": self.viewmats.unsqueeze(0).to(device=device, dtype=dtype),
            "Ks": self.Ks.unsqueeze(0).to(device=device, dtype=dtype),
        }


def load_cached_sample(cache_path: str | Path) -> CachedSample:
    cache_path = Path(cache_path)
    tensors_path = cache_path / "tensors.pt"
    metadata_path = cache_path / "metadata.json"
    if not tensors_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Cache is incomplete at {cache_path}; run Full_Duplex_Fix.preencode first"
        )
    tensors = torch.load(tensors_path, map_location="cpu", weights_only=True)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    sample = CachedSample(
        world_latents=tensors["world_latents"],
        prompt_embedding=tensors["prompt_embedding"],
        prompt_attention_mask=tensors["prompt_attention_mask"].bool(),
        negative_prompt_embedding=tensors["negative_prompt_embedding"],
        negative_prompt_attention_mask=tensors["negative_prompt_attention_mask"].bool(),
        viewmats=tensors["viewmats"],
        Ks=tensors["Ks"],
        metadata=metadata,
    )
    sample.validate()
    return sample
