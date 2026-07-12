from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from full_duplex.model import DuplexTurn
from full_duplex.training import FullDuplexTrainer, StepOutput


TEACHER_FORCED_REGIME = "teacher_forced_previous_gt_transition"


def previous_ground_truth_world_input(
    world_states: torch.Tensor,
    turn_index: int,
) -> torch.Tensor:
    """Return the exact teacher-forced input for transition ``t -> t+1``.

    Cached states are ``[state, C, H, W]``. Turn zero deliberately receives a
    batch-shaped all-zero state plus the model's existing NULL marker. Every
    later turn receives cached ground-truth state ``t``; no model prediction is
    substituted into this input stream.
    """

    if world_states.ndim != 4:
        raise ValueError(f"Expected cached [state,C,H,W], got {tuple(world_states.shape)}")
    if not 0 <= turn_index < world_states.shape[0] - 1:
        raise IndexError(f"Turn {turn_index} has no next-state target")
    if turn_index == 0:
        return torch.zeros_like(world_states[1:2]).unsqueeze(0)
    return world_states[turn_index : turn_index + 1].unsqueeze(0)


class TeacherForcedTransitionTrainer(FullDuplexTrainer):
    """Previous-GT -> next-GT trainer with bounded per-turn autograd graphs.

    One optimizer update still covers every configured transition. Each turn's
    loss is divided by the turn count and backpropagated immediately, so
    gradients accumulate exactly as a mean of independent transition losses
    while that turn's ten-denoise-step graph can be freed before the next turn.
    Historical predicted output *values* remain visible but are detached.
    """

    def __init__(self, config: dict[str, Any], mode: str, run_name: str):
        if config.get("training_regime") != TEACHER_FORCED_REGIME:
            raise ValueError(
                f"TeacherForcedTransitionTrainer requires training_regime={TEACHER_FORCED_REGIME}"
            )
        if not config.get("teacher_forced_world_inputs", False):
            raise ValueError("teacher_forced_world_inputs must be true")
        if not config.get("sequential_turn_backward", False):
            raise ValueError("sequential_turn_backward must be true")
        if float(config.get("teacher_forcing_ratio", 0.0)) != 1.0:
            raise ValueError("This strict teacher-forcing experiment requires teacher_forcing_ratio=1.0")
        if not config.get("detach_between_turns", False):
            raise ValueError(
                "detach_between_turns must be true because each transition is backpropagated "
                "and freed independently"
            )
        super().__init__(config, mode, run_name)

    def _num_teacher_forced_turns(self) -> int:
        num_turns = (
            1
            if self.mode == "single"
            else int(self.config.get("rollout_num_turns", self.config["num_micro_turns"]))
        )
        if not 1 <= num_turns <= self.config["num_micro_turns"]:
            raise ValueError("rollout_num_turns must fit the cached transition count")
        return num_turns

    def _camera_input(
        self,
        turn_index: int,
        camera_predictions: list[torch.Tensor],
    ) -> torch.Tensor:
        if turn_index == 0 or self.config.get("teacher_force_camera", False):
            return self.cameras[turn_index : turn_index + 1]
        return camera_predictions[-1]

    @staticmethod
    def _detached_history_turn(
        turn_index: int,
        world_input: torch.Tensor,
        camera_input: torch.Tensor,
        action_id: torch.Tensor,
        noise_input: torch.Tensor,
        world_output: torch.Tensor,
        camera_output: torch.Tensor,
    ) -> DuplexTurn:
        result = DuplexTurn(
            turn_index=turn_index,
            world_input=world_input.detach(),
            camera_input=camera_input.detach(),
            action_id=action_id.detach(),
            noise_input=noise_input.detach(),
            world_output=world_output.detach(),
            camera_output=camera_output.detach(),
        )
        tensors = (
            result.world_input,
            result.camera_input,
            result.noise_input,
            result.world_output,
            result.camera_output,
        )
        if any(tensor.grad_fn is not None or tensor.requires_grad for tensor in tensors):
            raise AssertionError("Teacher-forced history retained a cross-turn autograd edge")
        return result

    def forward_loss(self) -> StepOutput:
        """Fresh teacher-forced evaluation without optimizer/backward."""

        num_turns = self._num_teacher_forced_turns()
        history: list[DuplexTurn] = []
        predictions: list[torch.Tensor] = []
        camera_predictions: list[torch.Tensor] = []
        totals: list[torch.Tensor] = []
        flows: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        cameras: list[torch.Tensor] = []
        translations: list[torch.Tensor] = []
        rotations: list[torch.Tensor] = []
        intrinsics: list[torch.Tensor] = []
        per_turn: list[dict[str, float]] = []
        sequence_lengths: list[int] = []
        for turn_index in range(num_turns):
            self.model.clear_step_cache()
            world_input = previous_ground_truth_world_input(self.world_states, turn_index)
            camera_input = self._camera_input(turn_index, camera_predictions)
            target_state = self.world_states[turn_index + 1 : turn_index + 2].unsqueeze(0)
            target_camera = self.cameras[turn_index + 1 : turn_index + 2]
            result = self._denoise_turn(
                turn_index,
                world_input,
                camera_input,
                history,
                target_state,
                target_camera,
            )
            prediction = result.state.detach()
            camera_prediction = result.camera.detach()
            predictions.append(prediction)
            camera_predictions.append(camera_prediction)
            history.append(
                self._detached_history_turn(
                    turn_index,
                    world_input,
                    camera_input,
                    self.action_ids[turn_index : turn_index + 1],
                    self.fixed_noise[turn_index : turn_index + 1],
                    prediction,
                    camera_prediction,
                )
            )
            totals.append(result.total_loss)
            flows.append(result.flow_loss)
            states.append(result.state_loss)
            cameras.append(result.camera_losses.total)
            translations.append(result.camera_losses.translation)
            rotations.append(result.camera_losses.rotation)
            intrinsics.append(result.camera_losses.intrinsics)
            sequence_lengths.extend(result.sequence_lengths)
            per_turn.append(
                {
                    "turn": float(turn_index),
                    "input_state_index": float(-1 if turn_index == 0 else turn_index),
                    "target_state_index": float(turn_index + 1),
                    "flow_loss": float(result.flow_loss.detach()),
                    "state_loss": float(result.state_loss.detach()),
                    "camera_loss": float(result.camera_losses.total.detach()),
                    "translation_loss": float(result.camera_losses.translation.detach()),
                    "rotation_loss": float(result.camera_losses.rotation.detach()),
                    "intrinsics_loss": float(result.camera_losses.intrinsics.detach()),
                }
            )
        return StepOutput(
            total_loss=torch.stack(totals).mean(),
            flow_loss=torch.stack(flows).mean(),
            state_loss=torch.stack(states).mean(),
            camera_loss=torch.stack(cameras).mean(),
            translation_loss=torch.stack(translations).mean(),
            rotation_loss=torch.stack(rotations).mean(),
            intrinsics_loss=torch.stack(intrinsics).mean(),
            predictions=predictions,
            camera_predictions=camera_predictions,
            per_turn=per_turn,
            sequence_lengths=sequence_lengths,
            early_turn_future_gradient_norm=None,
            early_turn_prediction_probe=None,
            early_turn_local_gradient=None,
        )

    def train_step(self) -> dict[str, Any]:
        """Accumulate independent per-turn gradients with one optimizer update."""

        self.last_step_output = None
        self.model.clear_step_cache()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        num_turns = self._num_teacher_forced_turns()
        history: list[DuplexTurn] = []
        predictions: list[torch.Tensor] = []
        camera_predictions: list[torch.Tensor] = []
        detached_losses: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "total",
                "flow",
                "state",
                "camera",
                "translation",
                "rotation",
                "intrinsics",
            )
        }
        per_turn: list[dict[str, float]] = []
        sequence_lengths: list[int] = []

        for turn_index in range(num_turns):
            # Drop every graph-bearing encoding from the previous transition.
            # Detached history values are re-encoded in the current graph.
            self.model.clear_step_cache()
            world_input = previous_ground_truth_world_input(self.world_states, turn_index)
            camera_input = self._camera_input(turn_index, camera_predictions)
            if world_input.grad_fn is not None or world_input.requires_grad:
                raise AssertionError("Ground-truth world input unexpectedly requires gradients")
            target_state = self.world_states[turn_index + 1 : turn_index + 2].unsqueeze(0)
            target_camera = self.cameras[turn_index + 1 : turn_index + 2]
            with torch.autocast(
                device_type="cuda",
                dtype=self.compute_dtype,
                enabled=self.compute_dtype == torch.bfloat16,
            ):
                result = self._denoise_turn(
                    turn_index,
                    world_input,
                    camera_input,
                    history,
                    target_state,
                    target_camera,
                )
            if not torch.isfinite(result.total_loss):
                raise FloatingPointError(
                    f"Non-finite teacher-forced loss at optimizer step "
                    f"{self.global_step}, turn {turn_index}"
                )
            (result.total_loss / num_turns).backward()

            prediction = result.state.detach()
            camera_prediction = result.camera.detach()
            predictions.append(prediction)
            camera_predictions.append(camera_prediction)
            history.append(
                self._detached_history_turn(
                    turn_index,
                    world_input,
                    camera_input,
                    self.action_ids[turn_index : turn_index + 1],
                    self.fixed_noise[turn_index : turn_index + 1],
                    prediction,
                    camera_prediction,
                )
            )
            values = {
                "total": result.total_loss,
                "flow": result.flow_loss,
                "state": result.state_loss,
                "camera": result.camera_losses.total,
                "translation": result.camera_losses.translation,
                "rotation": result.camera_losses.rotation,
                "intrinsics": result.camera_losses.intrinsics,
            }
            for name, value in values.items():
                detached_losses[name].append(value.detach())
            sequence_lengths.extend(result.sequence_lengths)
            per_turn.append(
                {
                    "turn": float(turn_index),
                    "input_state_index": float(-1 if turn_index == 0 else turn_index),
                    "target_state_index": float(turn_index + 1),
                    "flow_loss": float(result.flow_loss.detach()),
                    "state_loss": float(result.state_loss.detach()),
                    "camera_loss": float(result.camera_losses.total.detach()),
                    "translation_loss": float(result.camera_losses.translation.detach()),
                    "rotation_loss": float(result.camera_losses.rotation.detach()),
                    "intrinsics_loss": float(result.camera_losses.intrinsics.detach()),
                }
            )
            del result
            self.model.clear_step_cache()

        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Non-finite teacher-forced gradient: {name}")
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            self.config["max_grad_norm"],
        )
        parameter_norm = self._parameter_norm()
        self.optimizer.step()
        self.lr_scheduler.step()
        torch.cuda.synchronize()
        self.global_step += 1
        self.epoch += 1
        elapsed = time.perf_counter() - started

        means = {
            name: torch.stack(values).mean() for name, values in detached_losses.items()
        }
        output = StepOutput(
            total_loss=means["total"],
            flow_loss=means["flow"],
            state_loss=means["state"],
            camera_loss=means["camera"],
            translation_loss=means["translation"],
            rotation_loss=means["rotation"],
            intrinsics_loss=means["intrinsics"],
            predictions=predictions,
            camera_predictions=camera_predictions,
            per_turn=per_turn,
            sequence_lengths=sequence_lengths,
            early_turn_future_gradient_norm=None,
            early_turn_prediction_probe=None,
            early_turn_local_gradient=None,
        )
        metrics = {
            "step": self.global_step,
            "epoch": self.epoch,
            "total_loss": float(output.total_loss),
            "flow_loss": float(output.flow_loss),
            "state_loss": float(output.state_loss),
            "camera_loss": float(output.camera_loss),
            "translation_loss": float(output.translation_loss),
            "rotation_loss": float(output.rotation_loss),
            "intrinsics_loss": float(output.intrinsics_loss),
            "gradient_norm": float(gradient_norm),
            "parameter_norm": parameter_norm,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
            "elapsed_seconds": elapsed,
            "latent_prediction_mse": float(output.state_loss),
            "early_turn_future_gradient_norm": None,
            "min_sequence_length": min(sequence_lengths),
            "max_sequence_length": max(sequence_lengths),
            "teacher_forced_world_inputs": True,
            "teacher_forced_camera_inputs": bool(
                self.config.get("teacher_force_camera", False)
            ),
            "cross_turn_bptt": False,
            "sequential_turn_backward": True,
            "per_turn": per_turn,
        }
        self.loss_history.append(metrics)
        self.last_step_output = output
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(f"[teacher-forced train] {json.dumps(metrics, sort_keys=True)}", flush=True)
        return metrics

    def load_task_warm_start(self, path: str | Path) -> dict[str, Any]:
        checkpoint_path = Path(path).resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_state_format") != "trainable_delta_over_strict_base":
            raise RuntimeError("Teacher-forced warm-start requires a pre-LoRA task delta")
        if checkpoint["preprocessing_metadata_hash"] != self.preprocessing_metadata[
            "preprocessing_config_hash"
        ]:
            raise RuntimeError("Teacher-forced warm-start preprocessing mismatch")
        if checkpoint["base_checkpoint_identity"] != self.preprocessing_metadata["identities"][
            "base_checkpoint"
        ]:
            raise RuntimeError("Teacher-forced warm-start base checkpoint mismatch")
        if checkpoint["fixed_noise_sha256"] != self.fixed_noise_sha256:
            raise RuntimeError("Teacher-forced warm-start fixed-noise mismatch")
        state = checkpoint["model"]
        source_keys = set(checkpoint["model_keys"])
        required = set(self.model.new_parameter_names())
        if set(state) != source_keys or source_keys != required:
            raise RuntimeError(
                "Teacher-forced warm-start task-key mismatch: "
                f"missing={sorted(required - source_keys)}, extra={sorted(source_keys - required)}"
            )
        current = self.model.state_dict()
        for key, value in state.items():
            if key not in current or current[key].shape != value.shape:
                raise RuntimeError(f"Teacher-forced warm-start shape/key mismatch: {key}")
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite warm-start tensor: {key}")
        incompatible = self.model.load_state_dict(state, strict=False)
        missing, unexpected = list(incompatible.missing_keys), list(incompatible.unexpected_keys)
        allowed_missing = set(current) - source_keys
        if set(missing) != allowed_missing or unexpected:
            raise RuntimeError(
                f"Teacher-forced warm-start mismatch missing={missing}, unexpected={unexpected}"
            )
        source_config = checkpoint["training_config"]
        report = {
            "source_checkpoint": str(checkpoint_path),
            "source_global_step": int(checkpoint["global_step"]),
            "source_best_loss": float(checkpoint["best_loss"]),
            "loaded_task_keys": sorted(source_keys),
            "loaded_task_elements": int(sum(value.numel() for value in state.values())),
            "source_num_backbone_blocks": int(source_config["num_backbone_blocks"]),
            "current_num_backbone_blocks": int(self.config["num_backbone_blocks"]),
            "source_spatial_token_stride": int(source_config["spatial_token_stride"]),
            "current_spatial_token_stride": int(self.config["spatial_token_stride"]),
            "optimizer_state_inherited": False,
            "global_step_reset_to_zero": True,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }
        self.warm_start_report = report
        self._write_run_manifest()
        with (self.run_dir / "warm_start_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(
            f"[teacher-forced warm start] loaded {len(source_keys)} task keys; "
            f"missing({len(missing)})={missing}; unexpected={unexpected}",
            flush=True,
        )
        return report

    def train(
        self,
        max_steps: int,
        resume: str | Path | None = None,
        *,
        warm_start: str | Path | None = None,
        override_resume_learning_rate: bool = False,
    ) -> dict[str, Any]:
        if resume is not None and warm_start is not None:
            raise ValueError("resume and warm_start are mutually exclusive")
        if warm_start is not None:
            self.load_task_warm_start(warm_start)
        return super().train(
            max_steps=max_steps,
            resume=resume,
            override_resume_learning_rate=override_resume_learning_rate,
        )
