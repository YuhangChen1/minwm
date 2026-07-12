from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from full_duplex.flow import denoising_sigmas
from full_duplex.teacher_forcing_training import (
    TEACHER_FORCED_REGIME,
    TeacherForcedTransitionTrainer,
)
from full_duplex.training import FullDuplexTrainer, _atomic_torch_save


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a deterministic latent-space rollout from a saved Full-Duplex checkpoint"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--num-denoising-steps",
        type=int,
        help=(
            "Inference-only sigma-grid override applied after strict checkpoint reload; "
            "model weights and fixed initial noise are unchanged"
        ),
    )
    args = parser.parse_args()
    if args.num_denoising_steps is not None and args.num_denoising_steps < 1:
        raise ValueError("--num-denoising-steps must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(checkpoint["training_config"])
    mode = checkpoint["mode"]
    trained_num_denoising_steps = int(config["num_denoising_steps"])
    inference_num_denoising_steps = (
        int(args.num_denoising_steps)
        if args.num_denoising_steps is not None
        else trained_num_denoising_steps
    )
    run_name = (
        f"checkpoint_prediction_step_{checkpoint['global_step']:06d}_"
        f"denoise_{inference_num_denoising_steps}"
    )
    training_regime = config.get("training_regime", "autoregressive_rollout")
    if training_regime == TEACHER_FORCED_REGIME:
        trainer_class = TeacherForcedTransitionTrainer
    elif training_regime == "autoregressive_rollout":
        trainer_class = FullDuplexTrainer
    else:
        raise ValueError(f"Unsupported checkpoint training_regime: {training_regime}")
    trainer = trainer_class(config, mode=mode, run_name=run_name)
    trainer.load_checkpoint(args.checkpoint)
    # Keep checkpoint compatibility checks strict, then alter only the
    # inference integration grid. The trained modules and initial noise do not
    # depend on the number of Euler intervals.
    trainer.config["num_denoising_steps"] = inference_num_denoising_steps
    trainer.sigmas = denoising_sigmas(
        inference_num_denoising_steps,
        trainer.config["timestep_shift"],
        device=trainer.device,
    )
    trainer.model.clear_step_cache()
    trainer.model.eval()
    with torch.inference_mode(), torch.autocast(
        "cuda",
        dtype=trainer.compute_dtype,
        enabled=trainer.compute_dtype == torch.bfloat16,
    ):
        output = trainer.forward_loss()
    payload = {
        "states": torch.cat([value.cpu() for value in output.predictions], dim=1),
        "cameras": torch.cat([value.cpu() for value in output.camera_predictions], dim=0),
        "target_states": trainer.world_states[1 : len(output.predictions) + 1].cpu(),
        "target_cameras": trainer.cameras[1 : len(output.camera_predictions) + 1].cpu(),
    }
    _atomic_torch_save(payload, args.output)
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_global_step": checkpoint["global_step"],
        "checkpoint_best_step": checkpoint["best_step"],
        "training_regime": training_regime,
        "teacher_forced_world_inputs": bool(
            config.get("teacher_forced_world_inputs", False)
        ),
        "teacher_forced_camera_inputs": bool(config.get("teacher_force_camera", False)),
        "input_state_indices": (
            [-1, *range(1, len(output.predictions))]
            if training_regime == TEACHER_FORCED_REGIME
            else [-1, *range(len(output.predictions) - 1)]
        ),
        "target_state_indices": list(range(1, len(output.predictions) + 1)),
        "fixed_noise_sha256": trainer.fixed_noise_sha256,
        "trained_num_denoising_steps": trained_num_denoising_steps,
        "inference_num_denoising_steps": inference_num_denoising_steps,
        "inference_sigmas": [float(value) for value in trainer.sigmas.cpu()],
        "num_turns": len(output.predictions),
        "sequence_length_min": min(output.sequence_lengths),
        "sequence_length_max": max(output.sequence_lengths),
        "states_shape": list(payload["states"].shape),
        "cameras_shape": list(payload["cameras"].shape),
        "finite": all(torch.isfinite(value).all().item() for value in payload.values()),
        "rgb_decoder_used": False,
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
