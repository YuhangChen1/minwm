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

from .checkpoint import checkpoint_identity
from .config import config_hash, load_config
from .data import load_cached_sample, tensor_sha256


LATENT_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
LATENT_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)


def _sha256_file(path: Path, chunk_size: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: str | Path, hash_contents: bool) -> dict[str, Any]:
    path = Path(path).resolve()
    result: dict[str, Any] = {
        "path": str(path),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    }
    if hash_contents:
        result["sha256"] = _sha256_file(path)
    return result


def _directory_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    files = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        file_hash = _sha256_file(file_path)
        files.append({"path": relative, "size": file_path.stat().st_size, "sha256": file_hash})
        digest.update(relative.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
    return {"path": str(path), "sha256": digest.hexdigest(), "files": files}


def _read_sample_metadata(path: str | Path) -> tuple[str, str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("The smallest-data metadata must contain exactly one item")
    caption = payload[0].get("caption")
    pose_str = payload[0].get("pose_str")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("Missing non-empty caption")
    if pose_str != "right-8, a-11":
        raise ValueError(f"Unexpected pose_str: {pose_str!r}")
    return caption, pose_str


def _decode_video(config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    reader = decord.VideoReader(
        config["video_path"],
        width=int(config["target_width"]),
        height=int(config["target_height"]),
        num_threads=4,
    )
    frame_count = len(reader)
    fps = float(reader.get_avg_fps())
    if frame_count != config["source_frames"]:
        raise ValueError(f"Expected 77 frames, decoded {frame_count}")
    if abs(fps - float(config["source_fps"])) > 1e-3:
        raise ValueError(f"Expected {config['source_fps']} FPS, decoded {fps}")
    frames = reader.get_batch(list(range(frame_count))).asnumpy()
    pixels = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous().float()
    pixels = pixels.div_(127.5).sub_(1.0)
    return pixels, {
        "frame_count": frame_count,
        "fps": fps,
        "decoded_shape": list(frames.shape),
        "pixel_shape": list(pixels.shape),
        "pixel_dtype": str(pixels.dtype),
    }


def _validate_manifest(config: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    manifest_path = Path(config["action_manifest"])
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["source_num_frames"] != 77 or manifest["step_frames"] != 4:
        raise ValueError("Manifest must describe 77 source frames and four frames per action")
    actions = manifest["actions"]
    if len(actions) != 19:
        raise ValueError(f"Expected 19 actions, got {len(actions)}")

    names = []
    alignment = []
    expected_start = 1
    for index, item in enumerate(actions):
        frames = list(item["source_frames"])
        expected = list(range(expected_start, expected_start + 4))
        if frames != expected:
            raise ValueError(f"Action {index} frame mismatch: {frames} vs {expected}")
        clip_path = (manifest_path.parent / item["file"]).resolve()
        if not clip_path.is_file() or len(decord.VideoReader(str(clip_path), num_threads=1)) != 4:
            raise ValueError(f"Action clip is missing or not four frames: {clip_path}")
        name = str(item["action"])
        names.append(name)
        alignment.append(
            {
                "action_index": index,
                "action": name,
                "source_frames": frames,
                "source_clip": str(clip_path),
                "input_state": index,
                "target_state": index + 1,
                "input_camera": index,
                "target_camera": index + 1,
            }
        )
        expected_start += 4
    if names != ["right"] * 8 + ["a"] * 11:
        raise ValueError(f"Unexpected action order: {names}")
    if expected_start != 77:
        raise ValueError("Actions do not cover source frames 1..76")
    return names, alignment


@torch.inference_mode()
def _encode_vae(
    pixels: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    from wan.modules.vae import _video_vae

    dtype = torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
    vae = _video_vae(pretrained_path=config["vae_checkpoint"], z_dim=16)
    vae = vae.eval().requires_grad_(False).to(device=device, dtype=dtype)
    mean = torch.tensor(LATENT_MEAN, device=device, dtype=dtype)
    inverse_std = torch.tensor(LATENT_STD, device=device, dtype=dtype).reciprocal()
    pixels = pixels.unsqueeze(0).to(device=device, dtype=dtype)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        latent = vae.encode(pixels, [mean, inverse_std]).float()
    latent = latent.permute(0, 2, 1, 3, 4).contiguous()[0].cpu()
    del pixels, vae
    gc.collect()
    torch.cuda.empty_cache()
    if tuple(latent.shape) != (20, 16, 60, 104):
        raise ValueError(f"Unexpected Wan VAE output: {tuple(latent.shape)}")
    return latent


@torch.inference_mode()
def _encode_prompts(
    prompts: list[str],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    from wan.modules.t5 import umt5_xxl
    from wan.modules.tokenizers import HuggingfaceTokenizer

    dtype = torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
    encoder = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=dtype,
        device=device,
    ).eval().requires_grad_(False)
    state = torch.load(
        config["t5_checkpoint"], map_location="cpu", mmap=True, weights_only=True
    )
    incompatible = encoder.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected strict T5 mismatch: {incompatible}")
    del state
    tokenizer = HuggingfaceTokenizer(
        name=config["t5_tokenizer"],
        seq_len=512,
        clean="whitespace",
        local_files_only=True,
    )
    ids, mask = tokenizer(prompts, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    context = encoder(ids, mask)
    for index in range(len(prompts)):
        length = int(mask[index].gt(0).sum().item())
        context[index, length:] = 0
    result = context.cpu().contiguous()
    result_mask = mask.bool().cpu().contiguous()
    del encoder, context, ids, mask
    gc.collect()
    torch.cuda.empty_cache()
    return result, result_mask


def _camera_data(pose_str: str) -> tuple[torch.Tensor, torch.Tensor]:
    from Wan21.scripts.data_preprocessing.build_worldplaygen_lmdb import poses_from_pose_str
    from wan_utils.dataset import build_viewmats_and_Ks

    intrinsics, poses = poses_from_pose_str(pose_str)
    viewmats, Ks = build_viewmats_and_Ks(intrinsics, poses)
    viewmats_tensor = torch.from_numpy(viewmats).float()
    Ks_tensor = torch.from_numpy(Ks).float()
    if viewmats_tensor.shape != (20, 4, 4) or Ks_tensor.shape != (20, 3, 3):
        raise ValueError("Camera preprocessing must produce exactly 20 states")
    if not torch.equal(viewmats_tensor[0], torch.eye(4)):
        raise ValueError("First-frame-normalized viewmat C0 must be identity")
    return viewmats_tensor, Ks_tensor


def _cache_is_current(cache_path: Path, preprocessing_hash: str) -> bool:
    metadata_path = cache_path / "metadata.json"
    tensors_path = cache_path / "tensors.pt"
    if not metadata_path.is_file() or not tensors_path.is_file():
        return False
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return (
        metadata.get("preprocessing_hash") == preprocessing_hash
        and isinstance(metadata.get("tensor_sha256"), dict)
    )


def preencode(
    config: dict[str, Any],
    *,
    force: bool = False,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    project_root = Path(config["project_root"])
    for path in (project_root, project_root / "Wan21", project_root / "shared"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    caption, pose_str = _read_sample_metadata(config["metadata_path"])
    action_names, alignment = _validate_manifest(config)
    hash_large = bool(config["hash_large_preprocessing_files"])
    identities = {
        "video": _file_identity(config["video_path"], True),
        "metadata": _file_identity(config["metadata_path"], True),
        "action_manifest": _file_identity(config["action_manifest"], True),
        "vae": _file_identity(config["vae_checkpoint"], hash_large),
        "t5": _file_identity(config["t5_checkpoint"], hash_large),
        "tokenizer": _directory_identity(config["t5_tokenizer"]),
        "ar_generator": checkpoint_identity(config["base_checkpoint"], compute_hash=hash_large),
    }
    preprocessing_description = {
        "cache_version": config["cache_version"],
        "caption": caption,
        "negative_prompt": config["negative_prompt"],
        "pose_str": pose_str,
        "resolution": [config["target_height"], config["target_width"]],
        "latent_mean": LATENT_MEAN,
        "latent_std": LATENT_STD,
        "camera_builder": "poses_from_pose_str -> build_viewmats_and_Ks",
        "identities": identities,
    }
    preprocessing_hash = config_hash(preprocessing_description)
    cache_path = Path(config["cache_path"])
    cache_path.mkdir(parents=True, exist_ok=True)
    if not force and _cache_is_current(cache_path, preprocessing_hash):
        sample = load_cached_sample(cache_path)
        return sample.metadata

    pixels, video_metadata = _decode_video(config)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Real Wan VAE/UMT5 preencoding requires a CUDA device")
    world_latents = _encode_vae(pixels, config, device)
    del pixels
    prompt_embeddings, prompt_masks = _encode_prompts(
        [caption, str(config["negative_prompt"])], config, device
    )
    viewmats, Ks = _camera_data(pose_str)
    tensors = {
        "world_latents": world_latents.half(),
        "prompt_embedding": prompt_embeddings[0].to(torch.bfloat16),
        "prompt_attention_mask": prompt_masks[0],
        "negative_prompt_embedding": prompt_embeddings[1].to(torch.bfloat16),
        "negative_prompt_attention_mask": prompt_masks[1],
        "viewmats": viewmats,
        "Ks": Ks,
    }
    latent_float = tensors["world_latents"].float()
    metadata: dict[str, Any] = {
        "cache_version": config["cache_version"],
        "preprocessing_hash": preprocessing_hash,
        "caption": caption,
        "negative_prompt": config["negative_prompt"],
        "pose_str": pose_str,
        "camera_source": "repository-derived; no measured camera ground truth",
        "camera_convention": "OpenCV world-to-camera, first-frame normalized, source quaternion xyzw",
        "camera_action_semantics": {
            "right": "+3 degree yaw per state",
            "a": "local -X translation, 0.08 per state",
        },
        "video": video_metadata,
        "action_names": action_names,
        "action_alignment": alignment,
        "identities": identities,
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
        "tensor_dtypes": {name: str(tensor.dtype) for name, tensor in tensors.items()},
        "tensor_sha256": {name: tensor_sha256(tensor) for name, tensor in tensors.items()},
        "latent_stats": {
            "min": float(latent_float.min()),
            "max": float(latent_float.max()),
            "mean": float(latent_float.mean()),
            "std": float(latent_float.std()),
        },
        "positive_prompt_tokens": int(prompt_masks[0].sum()),
        "negative_prompt_tokens": int(prompt_masks[1].sum()),
        "runtime_seconds": time.perf_counter() - started,
    }
    tensor_tmp = cache_path / "tensors.pt.tmp"
    metadata_tmp = cache_path / "metadata.json.tmp"
    torch.save(tensors, tensor_tmp)
    with metadata_tmp.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tensor_tmp, cache_path / "tensors.pt")
    os.replace(metadata_tmp, cache_path / "metadata.json")

    reloaded = torch.load(cache_path / "tensors.pt", map_location="cpu", weights_only=True)
    unequal = [name for name, tensor in tensors.items() if not torch.equal(tensor, reloaded[name])]
    if unequal:
        raise AssertionError(f"Cache reload is not bitwise equal: {unequal}")
    load_cached_sample(cache_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Full_Duplex_Fix/configs/overfit.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    metadata = preencode(config, force=args.force, device=args.device)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
