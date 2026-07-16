from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import decord
import torch
import torch.nn.functional as F
import torchvision

from .config import load_config
from .preencode import LATENT_MEAN, LATENT_STD


def _stats(tensor: torch.Tensor) -> dict:
    value = tensor.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(value).all()),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std()),
    }


def _contact_sheet(frames: torch.Tensor, path: Path) -> None:
    indices = torch.linspace(0, frames.shape[0] - 1, 8).round().long()
    images = frames[indices].permute(0, 3, 1, 2).float().div(255)
    images = F.interpolate(images, size=(240, 416), mode="bilinear", align_corners=False)
    grid = torchvision.utils.make_grid(images, nrow=4, padding=2)
    torchvision.io.write_png(grid.mul(255).round().byte(), str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Full_Duplex_Fix/configs/overfit.yaml")
    parser.add_argument("--latents", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    payload = torch.load(args.latents, map_location="cpu", weights_only=True)
    states = payload["states"]
    if states.shape != (1, 20, 16, 60, 104) or not torch.isfinite(states).all():
        raise ValueError(f"Expected finite [1,20,16,60,104], got {states.shape}")

    project_root = Path(config["project_root"])
    if str(project_root / "Wan21") not in sys.path:
        sys.path.insert(0, str(project_root / "Wan21"))
    from wan.modules.vae import _video_vae

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = _video_vae(pretrained_path=config["vae_checkpoint"], z_dim=16)
    vae = vae.eval().requires_grad_(False).to(device=device, dtype=dtype)
    latent = states.permute(0, 2, 1, 3, 4).contiguous().to(device=device, dtype=dtype)
    mean = torch.tensor(LATENT_MEAN, device=device, dtype=dtype)
    inverse_std = torch.tensor(LATENT_STD, device=device, dtype=dtype).reciprocal()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        decoded = vae.decode(latent, [mean, inverse_std]).float().clamp(-1, 1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if decoded.shape != (1, 3, 77, 480, 832) or not torch.isfinite(decoded).all():
        raise RuntimeError(f"Unexpected decoded output: {decoded.shape}")
    decode_seconds = time.perf_counter() - started
    frames = decoded[0].permute(1, 2, 3, 0).add(1).mul(127.5).round().clamp(0, 255)
    frames = frames.byte().cpu().contiguous()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = output.with_name(f"{output.stem}_contact_sheet.png")
    torchvision.io.write_video(
        str(output),
        frames,
        fps=float(config["source_fps"]),
        video_codec="libx264",
        options={"crf": "18"},
    )
    _contact_sheet(frames, preview)
    encoded = decord.VideoReader(str(output), num_threads=2)
    encoded_fps = float(encoded.get_avg_fps())
    if len(encoded) != 77 or tuple(encoded[0].shape) != (480, 832, 3):
        raise RuntimeError("Encoded MP4 failed the 77-frame 480x832 readback audit")
    if abs(encoded_fps - float(config["source_fps"])) > 1e-3:
        raise RuntimeError(f"Encoded MP4 FPS mismatch: {encoded_fps}")
    manifest = {
        "latent_path": str(Path(args.latents).resolve()),
        "video_path": str(output),
        "contact_sheet": str(preview),
        "latent_stats": _stats(states),
        "decoded_stats": _stats(decoded.cpu()),
        "frame_count": 77,
        "encoded_readback_frame_count": len(encoded),
        "encoded_readback_fps": encoded_fps,
        "resolution": [480, 832],
        "fps": config["source_fps"],
        "decode_seconds": decode_seconds,
        "vae_frozen": True,
    }
    manifest_path = output.with_name(f"{output.stem}_decode.json")
    manifest["decode_manifest"] = str(manifest_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
