from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from full_duplex.camera import rotation_6d_to_matrix


def _state_tensors(payload: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = payload["states"].float()
    target = payload["target_states"].float()
    if prediction.ndim == 5 and prediction.shape[0] == 1:
        prediction = prediction.squeeze(0)
    if target.ndim == 5 and target.shape[0] == 1:
        target = target.squeeze(0)
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            f"Expected matching [turn,16,H,W] states, got {prediction.shape} and {target.shape}"
        )
    return prediction, target


def _camera_tensors(payload: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = payload["cameras"].float()
    target = payload["target_cameras"].float()
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[-1] != 13:
        raise ValueError(
            f"Expected matching [turn,13] cameras, got {prediction.shape} and {target.shape}"
        )
    return prediction, target


def _rotation_degrees(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_rotation = rotation_6d_to_matrix(prediction[..., 3:9])
    target_rotation = rotation_6d_to_matrix(target[..., 3:9])
    relative = pred_rotation @ target_rotation.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def evaluate(predictions_path: Path, checkpoint_path: Path | None = None) -> dict[str, Any]:
    payload = torch.load(predictions_path, map_location="cpu", weights_only=True)
    required = {"states", "cameras", "target_states", "target_cameras"}
    if set(payload) != required:
        raise KeyError(f"Prediction payload keys differ: {sorted(set(payload) ^ required)}")
    states, target_states = _state_tensors(payload)
    cameras, target_cameras = _camera_tensors(payload)
    if states.shape[0] != cameras.shape[0]:
        raise ValueError("State/camera turn counts differ")
    for name, tensor in (
        ("states", states),
        ("target_states", target_states),
        ("cameras", cameras),
        ("target_cameras", target_cameras),
    ):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite tensor in {name}")

    state_mse = (states - target_states).square().flatten(1).mean(1)
    state_cosine = F.cosine_similarity(states.flatten(1), target_states.flatten(1), dim=1)
    translation_l2 = (cameras[:, :3] - target_cameras[:, :3]).norm(dim=-1)
    rotation_degrees = _rotation_degrees(cameras, target_cameras)
    intrinsics_rmse = (cameras[:, 9:] - target_cameras[:, 9:]).square().mean(1).sqrt()

    per_turn = []
    for turn in range(states.shape[0]):
        per_turn.append(
            {
                "turn": turn,
                "state_mse": float(state_mse[turn]),
                "state_cosine": float(state_cosine[turn]),
                "camera_translation_l2": float(translation_l2[turn]),
                "camera_rotation_degrees": float(rotation_degrees[turn]),
                "camera_intrinsics_rmse": float(intrinsics_rmse[turn]),
            }
        )

    best_turn = int(torch.argmin(state_mse))
    worst_turn = int(torch.argmax(state_mse))
    zero_baseline_mse = float(target_states.square().mean())
    report: dict[str, Any] = {
        "predictions_path": str(predictions_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path else None,
        "num_turns": states.shape[0],
        "state_shape": list(states.shape),
        "camera_shape": list(cameras.shape),
        "overall_state_mse": float(state_mse.mean()),
        "zero_latent_baseline_mse": zero_baseline_mse,
        "state_mse_over_zero_baseline": float(state_mse.mean()) / zero_baseline_mse,
        "overall_latent_cosine_similarity": float(
            F.cosine_similarity(states.flatten()[None], target_states.flatten()[None]).item()
        ),
        "predicted_latent_mean": float(states.mean()),
        "predicted_latent_std": float(states.std()),
        "target_latent_mean": float(target_states.mean()),
        "target_latent_std": float(target_states.std()),
        "mean_camera_translation_l2": float(translation_l2.mean()),
        "mean_camera_rotation_degrees": float(rotation_degrees.mean()),
        "mean_camera_intrinsics_rmse": float(intrinsics_rmse.mean()),
        "best_turn": per_turn[best_turn],
        "worst_turn": per_turn[worst_turn],
        "per_turn": per_turn,
        "rgb_decoder_used": False,
    }
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        report["checkpoint_global_step"] = int(checkpoint["global_step"])
        report["checkpoint_best_step"] = int(checkpoint["best_step"])
        report["checkpoint_best_loss"] = float(checkpoint["best_loss"])
        report["checkpoint_model_state_format"] = checkpoint["model_state_format"]
        config = checkpoint["training_config"]
        stride = int(config["spatial_token_stride"])
        _, patch_height, patch_width = config["patch_size"]
        projected_height = math.ceil((target_states.shape[-2] // patch_height) / stride) * patch_height
        projected_width = math.ceil((target_states.shape[-1] // patch_width) / stride) * patch_width
        low_resolution = F.interpolate(
            target_states,
            size=(projected_height, projected_width),
            mode="bilinear",
            align_corners=False,
        )
        lowpass_target = F.interpolate(
            low_resolution,
            size=target_states.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        report["validation_spatial_token_stride"] = stride
        report["validation_reconstructed_grid"] = [projected_height, projected_width]
        report["stride_lowpass_projection_mse"] = float(
            F.mse_loss(lowpass_target, target_states)
        )
        report["stride_lowpass_projection_cosine"] = float(
            F.cosine_similarity(
                lowpass_target.flatten()[None], target_states.flatten()[None]
            ).item()
        )
    return report


def _write_csv(report: dict[str, Any], path: Path) -> None:
    fields = list(report["per_turn"][0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["per_turn"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.predictions.with_name("evaluation.json")
    report = evaluate(args.predictions, args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    csv_path = output.with_suffix(".csv")
    _write_csv(report, csv_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[evaluation] json={output.resolve()} csv={csv_path.resolve()}")


if __name__ == "__main__":
    main()
