from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision

from full_duplex.preencode import LATENT_MEAN, LATENT_STD, file_identity


def _tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    value = value.float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(torch.isfinite(value).all().item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
    }


def _write_preview(frames: torch.Tensor, path: Path) -> None:
    """Write an eight-frame contact sheet without changing the MP4 frames."""
    frame_count = frames.shape[0]
    indices = torch.linspace(0, frame_count - 1, steps=min(8, frame_count)).round().long()
    thumbnails = frames[indices].permute(0, 3, 1, 2).float().div_(255.0)
    thumbnails = F.interpolate(thumbnails, size=(240, 416), mode="bilinear", align_corners=False)
    grid = torchvision.utils.make_grid(thumbnails, nrow=4, padding=2)
    torchvision.io.write_png(grid.mul_(255).round_().clamp_(0, 255).byte(), str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode real Full-Duplex predicted latents with the frozen Wan2.1 VAE"
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--vae-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.fps <= 0:
        raise ValueError(f"fps must be positive, got {args.fps}")

    predictions_path = args.predictions.resolve()
    vae_path = args.vae_checkpoint.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path = output_path.with_name(f"{output_path.stem}_preview.png")
    manifest_path = output_path.with_suffix(".json")

    payload = torch.load(predictions_path, map_location="cpu", weights_only=True)
    if "states" not in payload:
        raise KeyError(f"{predictions_path} does not contain a 'states' tensor")
    states = payload["states"]
    if states.ndim != 5 or states.shape[0] != 1 or states.shape[2] != 16:
        raise ValueError(
            "Expected predicted states [B=1, T, C=16, H, W], "
            f"received {tuple(states.shape)}"
        )
    if not torch.isfinite(states).all():
        raise FloatingPointError("Predicted latent states contain NaN or Inf")

    prediction_manifest_path = predictions_path.with_suffix(".json")
    prediction_manifest = None
    if prediction_manifest_path.is_file():
        prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))

    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root / "Wan21"))
    from wan.modules.vae import _video_vae

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    vae = _video_vae(pretrained_path=str(vae_path), z_dim=16)
    vae = vae.eval().requires_grad_(False).to(device=device, dtype=dtype)
    load_seconds = time.perf_counter() - started

    normalized_latents = states.permute(0, 2, 1, 3, 4).contiguous().to(
        device=device, dtype=dtype
    )
    mean = torch.tensor(LATENT_MEAN, device=device, dtype=dtype)
    inv_std = torch.tensor(LATENT_STD, device=device, dtype=dtype).reciprocal()

    decode_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda" and dtype == torch.bfloat16,
    ):
        decoded = vae.decode(normalized_latents, [mean, inv_std]).float().clamp_(-1, 1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - decode_started
    if not torch.isfinite(decoded).all():
        raise FloatingPointError("Decoded RGB tensor contains NaN or Inf")

    expected_frames = 1 + 4 * (states.shape[1] - 1)
    if decoded.shape != (1, 3, expected_frames, states.shape[3] * 8, states.shape[4] * 8):
        raise RuntimeError(
            "Unexpected Wan VAE output shape: "
            f"got {tuple(decoded.shape)}, expected "
            f"{(1, 3, expected_frames, states.shape[3] * 8, states.shape[4] * 8)}"
        )

    frames = decoded[0].permute(1, 2, 3, 0).add(1.0).mul(127.5)
    frames = frames.round().clamp_(0, 255).byte().cpu().contiguous()
    decoded_stats = _tensor_stats(decoded.cpu())
    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )

    del decoded, normalized_latents, vae, mean, inv_std
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    encode_started = time.perf_counter()
    torchvision.io.write_video(
        str(output_path),
        frames,
        fps=args.fps,
        video_codec="libx264",
        options={"crf": str(args.crf)},
    )
    _write_preview(frames, preview_path)
    encode_seconds = time.perf_counter() - encode_started

    latent_metrics: dict[str, float] = {}
    target_states = payload.get("target_states")
    if target_states is not None:
        targets = target_states.float()
        if targets.ndim == 4:
            targets = targets.unsqueeze(0)
        if targets.shape != states.shape:
            raise ValueError(
                f"target_states shape {tuple(targets.shape)} does not match states {tuple(states.shape)}"
            )
        predictions = states.float()
        latent_metrics = {
            "mse": float(F.mse_loss(predictions, targets).item()),
            "cosine_similarity": float(
                F.cosine_similarity(predictions.flatten(), targets.flatten(), dim=0).item()
            ),
        }

    teacher_forced_world_inputs = bool(
        prediction_manifest
        and prediction_manifest.get("teacher_forced_world_inputs", False)
    )
    teacher_forced_camera_inputs = bool(
        prediction_manifest
        and prediction_manifest.get("teacher_forced_camera_inputs", False)
    )
    manifest = {
        "prediction_payload": str(predictions_path),
        "prediction_manifest": prediction_manifest,
        "ground_truth_used_for_prediction": (
            teacher_forced_world_inputs or teacher_forced_camera_inputs
        ),
        "ground_truth_world_inputs_used_for_prediction": teacher_forced_world_inputs,
        "ground_truth_camera_inputs_used_for_prediction": teacher_forced_camera_inputs,
        "ground_truth_current_output_visible_to_model": False,
        "ground_truth_decoded": False,
        "vae_checkpoint": file_identity(vae_path, hash_contents=True),
        "latent_normalization": {
            "channel_mean": list(LATENT_MEAN),
            "channel_std": list(LATENT_STD),
        },
        "predicted_latents": _tensor_stats(states),
        "latent_metrics_against_cached_ground_truth": latent_metrics,
        "decoded_rgb": decoded_stats,
        "temporal_decode_rule": "1 + 4 * (num_latent_states - 1)",
        "frame_count": int(frames.shape[0]),
        "resolution": [int(frames.shape[1]), int(frames.shape[2])],
        "fps": args.fps,
        "duration_seconds": float(frames.shape[0] / args.fps),
        "video_codec": "libx264",
        "crf": args.crf,
        "video_path": str(output_path),
        "preview_path": str(preview_path),
        "vae_load_seconds": load_seconds,
        "vae_decode_seconds": decode_seconds,
        "video_encode_seconds": encode_seconds,
        "peak_cuda_memory_bytes": peak_memory_bytes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
