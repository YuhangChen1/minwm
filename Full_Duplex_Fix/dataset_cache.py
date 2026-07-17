from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset


CACHE_MANIFEST_NAME = "cache_manifest.json"
DATASET_CACHE_VERSION = "full_duplex_dataset_cache_v1"
_VIDEO_DIRECTORY = re.compile(r"^(?P<index>\d{6})_(?P<pose>.+)$")


def normalize_pose_str(pose_str: str) -> str:
    """Convert `right-8, a-11` to the video-directory suffix `right8a11`."""
    return "".join(character for character in pose_str if character not in "-, \t\r\n")


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_aligned_input_manifest(
    path: str | Path,
    *,
    project_root: str | Path,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("The preencode input manifest must be a JSON list")
    if expected_count is not None and len(payload) != expected_count:
        raise ValueError(f"Expected {expected_count} samples, found {len(payload)}")

    root = Path(project_root).expanduser().resolve()
    samples: list[dict[str, Any]] = []
    seen_videos: set[Path] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise TypeError(f"Sample {index} must be a JSON object")
        caption = raw.get("caption")
        pose_str = raw.get("pose_str")
        video_value = raw.get("video_path")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Sample {index} has no non-empty caption")
        if not isinstance(pose_str, str) or not pose_str.strip():
            raise ValueError(f"Sample {index} has no non-empty pose_str")
        if not isinstance(video_value, str) or not video_value.strip():
            raise ValueError(f"Sample {index} has no video_path")

        video_path = Path(video_value).expanduser()
        if not video_path.is_absolute():
            video_path = root / video_path
        video_path = video_path.resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Sample {index} video does not exist: {video_path}")
        if video_path in seen_videos:
            raise ValueError(f"Duplicate video_path at sample {index}: {video_path}")
        seen_videos.add(video_path)

        match = _VIDEO_DIRECTORY.fullmatch(video_path.parent.name)
        if match is None:
            raise ValueError(
                f"Sample {index} parent must be NNNNNN_<normalized pose>: {video_path.parent.name}"
            )
        directory_index = int(match.group("index"))
        if directory_index != index:
            raise ValueError(
                f"Sample {index} points at directory index {directory_index}: {video_path}"
            )
        normalized_pose = normalize_pose_str(pose_str)
        if match.group("pose") != normalized_pose:
            raise ValueError(
                f"Sample {index} pose/video mismatch: {normalized_pose!r} != "
                f"{match.group('pose')!r}"
            )
        samples.append(
            {
                "index": index,
                "caption": caption.strip(),
                "pose_str": pose_str,
                "normalized_pose": normalized_pose,
                "video_path": str(video_path),
            }
        )
    return samples


def validate_preencoded_tensors(
    tensors: dict[str, Any],
    *,
    sample_index: int,
    check_values: bool = True,
) -> None:
    expected = {
        "world_latents": (20, 16, 60, 104),
        "prompt_embedding": (512, 4096),
        "prompt_attention_mask": (512,),
        "viewmats": (20, 4, 4),
        "Ks": (20, 3, 3),
    }
    if set(tensors) != set(expected):
        raise ValueError(
            f"Cache sample {sample_index} keys differ: {sorted(tensors)} vs {sorted(expected)}"
        )
    for name, shape in expected.items():
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            actual = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor)
            raise ValueError(f"Cache sample {sample_index} {name}: expected {shape}, got {actual}")
        if check_values and tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Cache sample {sample_index} {name} contains NaN/Inf")
    if check_values:
        mask = tensors["prompt_attention_mask"].bool()
        if torch.count_nonzero(tensors["prompt_embedding"][~mask]).item():
            raise ValueError(f"Cache sample {sample_index} has non-zero padded text embeddings")


def write_cache_manifest(
    cache_root: str | Path,
    *,
    input_manifest: str | Path,
    samples: Sequence[dict[str, Any]],
) -> Path:
    cache_root = Path(cache_root).expanduser().resolve()
    entries = []
    for sample in samples:
        index = int(sample["index"])
        cache_file = Path("samples") / f"{index:06d}.pt"
        absolute_cache_file = cache_root / cache_file
        if not absolute_cache_file.is_file():
            raise FileNotFoundError(f"Preencoded sample is missing: {absolute_cache_file}")
        tensors = torch.load(absolute_cache_file, map_location="cpu", weights_only=True)
        validate_preencoded_tensors(tensors, sample_index=index)
        entries.append(
            {
                "index": index,
                "caption": sample["caption"],
                "pose_str": sample["pose_str"],
                "video_path": sample["video_path"],
                "cache_file": cache_file.as_posix(),
            }
        )
    payload = {
        "cache_version": DATASET_CACHE_VERSION,
        "input_manifest": str(Path(input_manifest).expanduser().resolve()),
        "input_manifest_sha256": sha256_file(input_manifest),
        "sample_count": len(entries),
        "samples": entries,
    }
    path = cache_root / CACHE_MANIFEST_NAME
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
    return path


def deterministic_split(
    sample_count: int,
    *,
    validation_size: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0 < validation_size < sample_count:
        raise ValueError("validation_size must leave non-empty train and validation splits")
    order = torch.randperm(sample_count, generator=torch.Generator().manual_seed(seed)).tolist()
    validation = sorted(order[:validation_size])
    train = sorted(order[validation_size:])
    return train, validation


class PreencodedVideoDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        cache_root: str | Path,
        *,
        indices: Sequence[int] | None = None,
        expected_count: int | None = None,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        manifest_path = self.cache_root / CACHE_MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Dataset cache is incomplete: {manifest_path}; run preencode_dataset first"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("cache_version") != DATASET_CACHE_VERSION:
            raise ValueError(f"Unsupported dataset cache version: {manifest.get('cache_version')}")
        entries = manifest.get("samples")
        if not isinstance(entries, list) or len(entries) != manifest.get("sample_count"):
            raise ValueError("Invalid cache manifest sample list")
        if expected_count is not None and len(entries) != expected_count:
            raise ValueError(f"Expected {expected_count} cached samples, found {len(entries)}")
        selected = list(range(len(entries))) if indices is None else [int(i) for i in indices]
        if any(index < 0 or index >= len(entries) for index in selected):
            raise IndexError("Dataset split contains an out-of-range sample index")
        self.manifest = manifest
        self.entries = [entries[index] for index in selected]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, item: int) -> dict[str, Any]:
        entry = self.entries[item]
        sample_index = int(entry["index"])
        cache_path = (self.cache_root / entry["cache_file"]).resolve()
        if not cache_path.is_relative_to(self.cache_root):
            raise ValueError(f"Cache path escapes cache root: {cache_path}")
        tensors = torch.load(cache_path, map_location="cpu", weights_only=True)
        validate_preencoded_tensors(
            tensors, sample_index=sample_index, check_values=False
        )
        return {**tensors, "sample_index": sample_index}
