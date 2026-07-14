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


def _evaluation_config(
    training_config: dict,
    requested_mode: str,
) -> tuple[dict, str, type[FullDuplexTrainer]]:
    trained_regime = training_config.get("training_regime", "autoregressive_rollout")
    if requested_mode == "trained":
        evaluation_mode = (
            "teacher_forced" if trained_regime == TEACHER_FORCED_REGIME else "autonomous"
        )
    else:
        evaluation_mode = requested_mode
    config = dict(training_config)
    if evaluation_mode == "teacher_forced":
        config.update(
            training_regime=TEACHER_FORCED_REGIME,
            teacher_forced_world_inputs=True,
            sequential_turn_backward=True,
            teacher_forcing_ratio=1.0,
            detach_between_turns=True,
        )
        trainer_class: type[FullDuplexTrainer] = TeacherForcedTransitionTrainer
    elif evaluation_mode == "autonomous":
        config.update(
            training_regime="autoregressive_rollout",
            teacher_forced_world_inputs=False,
            sequential_turn_backward=False,
            teacher_forcing_ratio=0.0,
            detach_between_turns=False,
        )
        trainer_class = FullDuplexTrainer
    else:
        raise ValueError(f"Unsupported evaluation mode: {evaluation_mode}")
    return config, evaluation_mode, trainer_class


def _load_inference_model_state(
    trainer: FullDuplexTrainer,
    checkpoint: dict,
) -> None:
    """Strictly restore model weights without imposing the training regime.

    Teacher-forced and autonomous evaluation use identical parameters and only
    differ in the source of the next turn's world input. Optimizer/resume
    compatibility must therefore not prevent evaluating both protocols.
    """

    if checkpoint["preprocessing_metadata_hash"] != trainer.preprocessing_metadata[
        "preprocessing_config_hash"
    ]:
        raise RuntimeError("Inference checkpoint preprocessing metadata mismatch")
    if checkpoint["base_checkpoint_identity"] != trainer.preprocessing_metadata[
        "identities"
    ]["base_checkpoint"]:
        raise RuntimeError("Inference checkpoint strict-base identity mismatch")
    if checkpoint["fixed_noise_sha256"] != trainer.fixed_noise_sha256:
        raise RuntimeError("Inference checkpoint fixed-noise identity mismatch")

    state = checkpoint["model"]
    expected = set(checkpoint["model_keys"])
    if set(state) != expected:
        raise RuntimeError("Inference checkpoint model-key manifest mismatch")
    state_format = checkpoint["model_state_format"]
    if state_format == "full":
        incompatible = trainer.model.load_state_dict(state, strict=True)
        missing, unexpected = list(incompatible.missing_keys), list(incompatible.unexpected_keys)
    elif state_format == "trainable_delta_over_strict_base":
        current_trainable = {
            name
            for name, parameter in trainer.model.named_parameters()
            if parameter.requires_grad
        }
        if expected != current_trainable:
            raise RuntimeError(
                "Inference trainable-delta mismatch: "
                f"missing={sorted(current_trainable - expected)}, "
                f"extra={sorted(expected - current_trainable)}"
            )
        incompatible = trainer.model.load_state_dict(state, strict=False)
        missing, unexpected = list(incompatible.missing_keys), list(incompatible.unexpected_keys)
        allowed_missing = set(trainer.model.state_dict()) - expected
        if set(missing) != allowed_missing or unexpected:
            raise RuntimeError(
                f"Inference delta load mismatch missing={missing}, unexpected={unexpected}"
            )
    else:
        raise ValueError(
            "Dual-protocol prediction currently supports full and non-LoRA task-delta "
            f"checkpoints, got {state_format}"
        )
    print(
        f"[inference load] format={state_format} missing={len(missing)} "
        f"unexpected={len(unexpected)}",
        flush=True,
    )
    # Evaluation does not restore optimizer/RNG state, but step metadata is
    # still provenance and prevents any step-zero-only training diagnostics
    # from being mistaken for inference work.
    trainer.global_step = int(checkpoint["global_step"])
    trainer.epoch = int(checkpoint["epoch"])
    trainer.best_step = int(checkpoint["best_step"])
    trainer.best_loss = float(checkpoint["best_loss"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a deterministic latent-space rollout from a saved Full-Duplex checkpoint"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--evaluation-mode",
        choices=("trained", "teacher_forced", "autonomous"),
        default="trained",
        help=(
            "World-input protocol used for this fresh prediction. 'trained' preserves "
            "legacy behavior; the explicit modes allow both metrics from one checkpoint"
        ),
    )
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
    training_config = dict(checkpoint["training_config"])
    mode = checkpoint["mode"]
    trained_num_denoising_steps = int(training_config["num_denoising_steps"])
    inference_num_denoising_steps = (
        int(args.num_denoising_steps)
        if args.num_denoising_steps is not None
        else trained_num_denoising_steps
    )
    run_name = (
        f"checkpoint_prediction_step_{checkpoint['global_step']:06d}_"
        f"denoise_{inference_num_denoising_steps}"
    )
    trained_regime = training_config.get("training_regime", "autoregressive_rollout")
    config, evaluation_mode, trainer_class = _evaluation_config(
        training_config, args.evaluation_mode
    )
    trainer = trainer_class(config, mode=mode, run_name=run_name)
    _load_inference_model_state(trainer, checkpoint)
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
        "checkpoint_training_regime": trained_regime,
        "evaluation_mode": evaluation_mode,
        "teacher_forced_world_inputs": evaluation_mode == "teacher_forced",
        "teacher_forced_camera_inputs": bool(config.get("teacher_force_camera", False)),
        "input_state_indices": (
            [-1, *range(1, len(output.predictions))]
            if evaluation_mode == "teacher_forced"
            else [None] * len(output.predictions)
        ),
        "input_state_sources": (
            ["null", *(["previous_ground_truth"] * (len(output.predictions) - 1))]
            if evaluation_mode == "teacher_forced"
            else ["null", *(["previous_prediction"] * (len(output.predictions) - 1))]
        ),
        "ground_truth_world_inputs_used_for_prediction": (
            evaluation_mode == "teacher_forced"
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
