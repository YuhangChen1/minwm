from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import decord
import torch

from full_duplex.camera import CAMERA_REPRESENTATION, viewmats_and_Ks_to_camera
from full_duplex.config import config_hash, load_config


LATENT_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
LATENT_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)


def sha256_file(path: str | Path, chunk_size: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path, *, hash_contents: bool = True) -> dict[str, Any]:
    path = Path(path).resolve()
    stat = path.stat()
    identity: dict[str, Any] = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_contents:
        identity["sha256"] = sha256_file(path)
    return identity


def _video_frames(path: str, target_height: int, target_width: int) -> tuple[torch.Tensor, dict[str, Any]]:
    reader = decord.VideoReader(path, width=target_width, height=target_height, num_threads=4)
    count = len(reader)
    fps = float(reader.get_avg_fps())
    frames = reader.get_batch(list(range(count))).asnumpy()
    tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous().float()
    tensor = tensor.div_(127.5).sub_(1.0)
    return tensor, {
        "original_frame_count": count,
        "fps": fps,
        "decoded_shape": list(frames.shape),
        "pixel_tensor_shape": list(tensor.shape),
        "pixel_dtype": str(tensor.dtype),
    }


def _load_actions(config: dict[str, Any], rgb_count: int) -> tuple[list[str], torch.Tensor, list[dict[str, Any]], str]:
    manifest_path = Path(config["action_manifest"])
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    actions = manifest["actions"]
    if manifest["source_num_frames"] != rgb_count:
        raise ValueError(
            f"Manifest says {manifest['source_num_frames']} frames, decoded video has {rgb_count}"
        )
    if len(actions) != config["num_micro_turns"]:
        raise ValueError(f"Manifest has {len(actions)} actions, config has {config['num_micro_turns']} turns")

    names: list[str] = []
    ranges: list[dict[str, Any]] = []
    manifest_root = manifest_path.parent
    expected_start = 1
    for index, action in enumerate(actions):
        frames = action.get("source_frames")
        if frames is None:
            frames = list(range(action["source_frame_start"], action["source_frame_end"] + 1))
        if frames != list(range(expected_start, expected_start + config["num_frame_per_turn"])):
            raise ValueError(f"Non-contiguous or mis-sized action {index}: {frames}")
        expected_start += config["num_frame_per_turn"]
        clip_path = (manifest_root / action["file"]).resolve()
        clip_reader = decord.VideoReader(str(clip_path), num_threads=1)
        if len(clip_reader) != config["num_frame_per_turn"]:
            raise ValueError(f"{clip_path} contains {len(clip_reader)} frames")
        name = str(action["action"])
        names.append(name)
        ranges.append(
            {
                "turn": index,
                "action": name,
                "source_frame_start": frames[0],
                "source_frame_end": frames[-1],
                "num_rgb_frames": len(frames),
                "clip_path": str(clip_path),
                "input_state_index": index,
                "target_state_index": index + 1,
                "input_camera_index": index,
                "target_camera_index": index + 1,
                "model_input_state": "NULL_ZERO" if index == 0 else f"predicted_state_{index}",
            }
        )
    if expected_start != rgb_count:
        raise ValueError(f"Actions end at frame {expected_start - 1}, expected {rgb_count - 1}")

    vocabulary = list(config["action_vocabulary"])
    if "NO_OP" not in vocabulary:
        raise ValueError("action_vocabulary must reserve NO_OP")
    unknown = sorted(set(names) - set(vocabulary))
    if unknown:
        raise ValueError(f"Actions absent from vocabulary: {unknown}")
    ids = torch.tensor([vocabulary.index(name) for name in names], dtype=torch.long)
    pose_str = f"right-{names.count('right')}, a-{names.count('a')}"
    return names, ids, ranges, pose_str


@torch.inference_mode()
def _encode_vae(pixel: torch.Tensor, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    sys.path.insert(0, str(Path(config["project_root"]) / "Wan21"))
    from wan.modules.vae import _video_vae

    model = _video_vae(pretrained_path=config["vae_checkpoint"], z_dim=16)
    dtype = torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
    model = model.eval().requires_grad_(False).to(device=device, dtype=dtype)
    pixel = pixel.unsqueeze(0).to(device=device, dtype=dtype)
    mean = torch.tensor(LATENT_MEAN, device=device, dtype=dtype)
    inv_std = torch.tensor(LATENT_STD, device=device, dtype=dtype).reciprocal()
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
        latent = model.encode(pixel, [mean, inv_std]).float()
    latent = latent.permute(0, 2, 1, 3, 4).contiguous().cpu()[0]
    del pixel, model
    gc.collect()
    torch.cuda.empty_cache()
    return latent


@torch.inference_mode()
def _encode_text(prompt: str, config: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sys.path.insert(0, str(Path(config["project_root"]) / "Wan21"))
    from wan.modules.t5 import umt5_xxl
    from wan.modules.tokenizers import HuggingfaceTokenizer

    dtype = torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
    model = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=dtype,
        device=device,
    ).eval().requires_grad_(False)
    state = torch.load(config["t5_checkpoint"], map_location="cpu", mmap=True, weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected strict T5 mismatch: {incompatible}")
    del state
    tokenizer = HuggingfaceTokenizer(
        name=config["t5_tokenizer"], seq_len=512, clean="whitespace", local_files_only=True
    )
    ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
    ids, mask = ids.to(device), mask.to(device)
    context = model(ids, mask)
    seq_len = int(mask.gt(0).sum().item())
    context[:, seq_len:] = 0
    result = context[0].cpu().contiguous()
    result_mask = mask[0].bool().cpu().contiguous()
    del model, context, ids, mask
    gc.collect()
    torch.cuda.empty_cache()
    return result, result_mask


def _camera_data(config: dict[str, Any], pose_str: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sys.path.insert(0, str(Path(config["project_root"])))
    sys.path.insert(0, str(Path(config["project_root"]) / "Wan21"))
    from Wan21.scripts.data_preprocessing.build_worldplaygen_lmdb import poses_from_pose_str
    from wan_utils.dataset import build_viewmats_and_Ks

    intrinsics, poses = poses_from_pose_str(pose_str)
    viewmats_np, Ks_np = build_viewmats_and_Ks(intrinsics, poses)
    viewmats = torch.from_numpy(viewmats_np).float()
    Ks = torch.from_numpy(Ks_np).float()
    camera = viewmats_and_Ks_to_camera(viewmats, Ks)
    return viewmats, Ks, camera


def _validate_tensors(tensors: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, Any]:
    for name, tensor in tensors.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite cache tensor: {name}")
    latent = tensors["world_state_latents"]
    expected = (
        config["num_micro_turns"] + 1,
        config["latent_channels"],
        config["latent_height"],
        config["latent_width"],
    )
    if tuple(latent.shape) != expected:
        raise ValueError(f"Expected latent {expected}, got {tuple(latent.shape)}")
    if tensors["camera"].shape != (expected[0], config["camera_dim"]):
        raise ValueError(f"Unexpected camera shape {tuple(tensors['camera'].shape)}")
    if tensors["action_ids"].numel() != config["num_micro_turns"]:
        raise ValueError("action/state alignment length mismatch")
    prompt = tensors["prompt_embedding"]
    prompt_mask = tensors["prompt_attention_mask"]
    if prompt.shape != (512, 4096) or prompt_mask.shape != (512,):
        raise ValueError(f"Unexpected prompt cache shapes {prompt.shape}, {prompt_mask.shape}")
    if torch.count_nonzero(prompt[~prompt_mask]).item() != 0:
        raise ValueError("T5 padding embeddings were not zeroed")

    translation_delta = tensors["camera"][1:, :3] - tensors["camera"][:-1, :3]
    return {
        "latent_shape": list(latent.shape),
        "latent_dtype": str(latent.dtype),
        "latent_min": float(latent.min()),
        "latent_max": float(latent.max()),
        "latent_mean": float(latent.float().mean()),
        "latent_std": float(latent.float().std()),
        "prompt_embedding_shape": list(prompt.shape),
        "prompt_embedding_dtype": str(prompt.dtype),
        "prompt_nonpadding_tokens": int(prompt_mask.sum()),
        "camera_shape": list(tensors["camera"].shape),
        "camera_dtype": str(tensors["camera"].dtype),
        "camera_max_translation_step": float(translation_delta.norm(dim=-1).max()),
        "action_ids_shape": list(tensors["action_ids"].shape),
    }


def _cache_matches(cache_dir: Path, expected_hash: str) -> bool:
    metadata_path = cache_dir / "metadata.json"
    tensor_path = cache_dir / "tensors.pt"
    if not metadata_path.is_file() or not tensor_path.is_file():
        return False
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return metadata.get("preprocessing_config_hash") == expected_hash


def preencode(config: dict[str, Any], force: bool = False) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    started = time.perf_counter()
    cache_dir = Path(config["cache_path"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("[preencode] hashing VAE/T5/source identities", flush=True)
    identities = {
        "source_video": file_identity(config["video_path"]),
        "action_manifest": file_identity(config["action_manifest"]),
        "vae_checkpoint": file_identity(config["vae_checkpoint"]),
        "t5_checkpoint": file_identity(config["t5_checkpoint"]),
        "base_checkpoint": file_identity(config["base_checkpoint"], hash_contents=False),
    }
    preprocessing_fields = {
        "cache_version": config["cache_version"],
        "target_height": config["target_height"],
        "target_width": config["target_width"],
        "num_frame_per_turn": config["num_frame_per_turn"],
        "num_micro_turns": config["num_micro_turns"],
        "prompt": config["prompt"],
        "latent_mean": LATENT_MEAN,
        "latent_std": LATENT_STD,
        "identities": identities,
    }
    preprocessing_hash = config_hash(preprocessing_fields)

    if not force and _cache_matches(cache_dir, preprocessing_hash):
        tensors = torch.load(cache_dir / "tensors.pt", map_location="cpu", weights_only=True)
        with (cache_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        stats = _validate_tensors(tensors, config)
        print(f"[preencode] valid cache hit {cache_dir}: {stats}")
        return tensors, metadata

    pixel, video_metadata = _video_frames(
        config["video_path"], config["target_height"], config["target_width"]
    )
    names, action_ids, ranges, pose_str = _load_actions(
        config, video_metadata["original_frame_count"]
    )
    print(f"[preencode] pixel video {tuple(pixel.shape)} {pixel.dtype}")
    print(f"[preencode] actions={len(names)} labels={names}")
    device = torch.device("cuda:0")
    latent = _encode_vae(pixel, config, device)
    del pixel
    print(f"[preencode] VAE latent {tuple(latent.shape)} {latent.dtype}")
    prompt_embedding, prompt_mask = _encode_text(config["prompt"], config, device)
    print(f"[preencode] T5 {tuple(prompt_embedding.shape)} {prompt_embedding.dtype}")
    viewmats, Ks, camera = _camera_data(config, pose_str)

    tensors = {
        "world_state_latents": latent.half(),
        "prompt_embedding": prompt_embedding.to(torch.bfloat16),
        "prompt_attention_mask": prompt_mask,
        "viewmats": viewmats,
        "Ks": Ks,
        "camera": camera,
        "action_ids": action_ids,
    }
    stats = _validate_tensors(tensors, config)
    metadata: dict[str, Any] = {
        "cache_version": config["cache_version"],
        "preprocessing_config_hash": preprocessing_hash,
        "sample_id": Path(config["dataset_path"]).name,
        "source_video_path": config["video_path"],
        "prompt": config["prompt"],
        "video": video_metadata,
        "micro_turn_duration_ms": (
            1000.0 * config["num_frame_per_turn"] / video_metadata["fps"]
        ),
        "diagram_nominal_micro_turn_ms": 200.0,
        "world_state": {
            "latent_frame_count": int(latent.shape[0]),
            "normalization": "(mu - channel_mean) / channel_std",
            "channel_mean": list(LATENT_MEAN),
            "channel_std": list(LATENT_STD),
            "initial_state_index": 0,
            "initial_state_belongs_to_action_0": False,
            "turn_0_model_input": "all-zero latent plus NULL_WORLD_STATE token",
            "turn_targets": "turn t predicts cached latent t+1",
        },
        "camera_representation": CAMERA_REPRESENTATION,
        "camera_coordinate_system": "OpenCV world-to-camera (w2c), xyzw quaternion at source",
        "camera_source": (
            "No measured camera file exists in this minimal sample; deterministically generated "
            "from the real ordered action labels with the repository's poses_from_pose_str() "
            "and build_viewmats_and_Ks() utilities"
        ),
        "camera_pose_spec": pose_str,
        "camera_action_relation": "right controls rotation; a controls translation in the repository trajectory utility",
        "action_vocabulary": list(config["action_vocabulary"]),
        "action_names": names,
        "action_to_state_alignment": ranges,
        "micro_turn_boundaries": ranges,
        "num_micro_turns": len(names),
        "identities": identities,
        "stats": stats,
        "runtime_seconds": time.perf_counter() - started,
    }
    tensor_tmp = cache_dir / "tensors.pt.tmp"
    metadata_tmp = cache_dir / "metadata.json.tmp"
    torch.save(tensors, tensor_tmp)
    with metadata_tmp.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tensor_tmp, cache_dir / "tensors.pt")
    os.replace(metadata_tmp, cache_dir / "metadata.json")

    reloaded = torch.load(cache_dir / "tensors.pt", map_location="cpu", weights_only=True)
    unequal = [name for name in tensors if not torch.equal(tensors[name], reloaded[name])]
    if unequal:
        raise AssertionError(f"Cache reload was not bitwise equal: {unequal}")
    print(f"[preencode] wrote bitwise-stable cache {cache_dir}")
    print(f"[preencode] stats {json.dumps(stats, sort_keys=True)}")
    return reloaded, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full_duplex/configs/overfit.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    preencode(config, force=args.force)


if __name__ == "__main__":
    main()
