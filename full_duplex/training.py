from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from full_duplex.camera import CAMERA_REPRESENTATION, CameraLosses, camera_loss
from full_duplex.flow import denoising_sigmas, flow_step, flow_target
from full_duplex.lora import assert_zero_lora_residual
from full_duplex.model import DuplexTurn, FullDuplexWanModel


@dataclass
class DenoisedTurn:
    state: torch.Tensor
    camera: torch.Tensor
    flow_loss: torch.Tensor
    state_loss: torch.Tensor
    camera_losses: CameraLosses
    total_loss: torch.Tensor
    sequence_lengths: list[int]


@dataclass
class StepOutput:
    total_loss: torch.Tensor
    flow_loss: torch.Tensor
    state_loss: torch.Tensor
    camera_loss: torch.Tensor
    translation_loss: torch.Tensor
    rotation_loss: torch.Tensor
    intrinsics_loss: torch.Tensor
    predictions: list[torch.Tensor]
    camera_predictions: list[torch.Tensor]
    per_turn: list[dict[str, float]]
    sequence_lengths: list[int]
    early_turn_future_gradient_norm: float | None
    early_turn_prediction_probe: torch.Tensor | None
    early_turn_local_gradient: torch.Tensor | None


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.view(torch.uint16)
    array = cpu.numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_symlink(target: Path, link: Path) -> None:
    temporary = link.with_name(link.name + ".tmp")
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    temporary.symlink_to(target.name)
    os.replace(temporary, link)


class FullDuplexTrainer:
    def __init__(self, config: dict[str, Any], mode: str, run_name: str):
        if mode not in ("single", "rollout"):
            raise ValueError("mode must be 'single' or 'rollout'")
        self.config = dict(config)
        self.mode = mode
        self.run_name = run_name
        self.device = torch.device("cuda:0")
        self.compute_dtype = (
            torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
        )
        self.run_dir = Path(config["output_dir"]) / run_name
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        if config["batch_size"] != 1:
            raise ValueError("The minimal real-sample overfit trainer currently requires batch_size: 1")
        if config["gradient_accumulation_steps"] != 1:
            raise ValueError(
                "Cross-turn graph accumulation is not implemented; use gradient_accumulation_steps: 1"
            )

        self._seed_everything(config["seed"])
        cache_dir = Path(config["cache_path"])
        with (cache_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            self.preprocessing_metadata = json.load(handle)
        self._validate_cache_identity()
        self.cache = torch.load(cache_dir / "tensors.pt", map_location="cpu", weights_only=True)
        self.world_states = self.cache["world_state_latents"].to(
            self.device, dtype=self.compute_dtype
        )
        self.cameras = self.cache["camera"].to(self.device, dtype=self.compute_dtype)
        self.action_ids = self.cache["action_ids"].to(self.device)
        self.prompt_embedding = self.cache["prompt_embedding"].unsqueeze(0).to(
            self.device, dtype=self.compute_dtype
        )
        self.prompt_attention_mask = self.cache["prompt_attention_mask"].unsqueeze(0).to(
            self.device
        )

        model = FullDuplexWanModel.from_checkpoint(config)
        self.parameter_counts = model.configure_trainable_parameters(
            config["train_backbone"],
            config.get("train_base_world_head", False),
        )
        # Keep master parameters fp32; autocast performs Wan compute in bf16.
        self.model = model.to(self.device)
        named_trainable = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if not named_trainable:
            raise RuntimeError("No trainable parameters")
        world_head_prefixes = (
            "backbone.head.",
            "world_residual_head.",
            "world_residual_norm.",
        )
        world_prior_prefix = "world_time_space_prior."
        world_head_parameters = [
            parameter
            for name, parameter in named_trainable
            if name.startswith(world_head_prefixes)
        ]
        world_prior_parameters = [
            parameter
            for name, parameter in named_trainable
            if name.startswith(world_prior_prefix)
        ]
        lora_parameters = [
            parameter
            for name, parameter in named_trainable
            if name.endswith(".lora_A") or name.endswith(".lora_B")
        ]
        other_parameters = [
            parameter
            for name, parameter in named_trainable
            if not name.startswith(world_head_prefixes)
            and not name.startswith(world_prior_prefix)
            and not (name.endswith(".lora_A") or name.endswith(".lora_B"))
        ]
        world_head_multiplier = float(
            config.get("world_head_learning_rate_multiplier", 1.0)
        )
        world_prior_multiplier = float(
            config.get("world_prior_learning_rate_multiplier", 1.0)
        )
        lora_multiplier = float(config.get("lora_learning_rate_multiplier", 1.0))
        if (
            world_head_multiplier <= 0
            or world_prior_multiplier <= 0
            or lora_multiplier <= 0
        ):
            raise ValueError("World and LoRA LR multipliers must be positive")
        if world_head_multiplier != 1.0 and not world_head_parameters:
            raise ValueError(
                "A world-head LR multiplier requires world_residual_head or a trainable base head"
            )
        if world_prior_multiplier != 1.0 and not world_prior_parameters:
            raise ValueError(
                "A world-prior LR multiplier requires world_time_space_prior: true"
            )
        if lora_multiplier != 1.0 and not lora_parameters:
            raise ValueError("A LoRA LR multiplier requires lora_enabled: true")
        default_parameters = list(other_parameters)
        if world_head_multiplier == 1.0:
            default_parameters.extend(world_head_parameters)
        if world_prior_multiplier == 1.0:
            default_parameters.extend(world_prior_parameters)
        if lora_multiplier == 1.0:
            default_parameters.extend(lora_parameters)
        optimizer_groups = []
        if default_parameters:
            optimizer_groups.append(
                {
                    "params": default_parameters,
                    "lr": config["learning_rate"],
                    "group_name": "default",
                }
            )
        if world_head_multiplier != 1.0:
            optimizer_groups.append(
                {
                    "params": world_head_parameters,
                    "lr": config["learning_rate"] * world_head_multiplier,
                    "group_name": "world_head",
                }
            )
        if world_prior_multiplier != 1.0:
            optimizer_groups.append(
                {
                    "params": world_prior_parameters,
                    "lr": config["learning_rate"] * world_prior_multiplier,
                    "group_name": "world_prior",
                }
            )
        if lora_multiplier != 1.0:
            optimizer_groups.append(
                {
                    "params": lora_parameters,
                    "lr": config["learning_rate"] * lora_multiplier,
                    "group_name": "lora",
                }
            )
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.999),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _step: 1.0)
        self.sigmas = denoising_sigmas(
            config["num_denoising_steps"],
            config["timestep_shift"],
            device=self.device,
        )
        self.fixed_noise = self._make_fixed_noise()
        self.fixed_noise_sha256 = _tensor_sha256(self.fixed_noise)
        self.global_step = 0
        self.epoch = 0
        self.best_loss = math.inf
        self.best_step = -1
        self.loss_history: list[dict[str, Any]] = []
        self.last_step_output: StepOutput | None = None
        self.warm_start_report: dict[str, Any] | None = None
        self._write_run_manifest()

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _validate_cache_identity(self) -> None:
        metadata = self.preprocessing_metadata
        if metadata.get("cache_version") != self.config["cache_version"]:
            raise RuntimeError("Cache version mismatch; rerun preencode.py")
        if metadata.get("num_micro_turns") != self.config["num_micro_turns"]:
            raise RuntimeError("Cache/config micro-turn mismatch")
        for name, path_key in (
            ("source_video", "video_path"),
            ("action_manifest", "action_manifest"),
            ("base_checkpoint", "base_checkpoint"),
            ("vae_checkpoint", "vae_checkpoint"),
            ("t5_checkpoint", "t5_checkpoint"),
        ):
            cached = metadata["identities"][name]
            current = Path(self.config[path_key]).stat()
            if cached["size"] != current.st_size or cached["mtime_ns"] != current.st_mtime_ns:
                raise RuntimeError(f"{name} changed; rerun preencode.py")

    def _make_fixed_noise(self) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(self.config["fixed_noise_seed"])
        shape = (
            self.config["num_micro_turns"],
            1,
            self.config["latent_channels"],
            self.config["latent_height"],
            self.config["latent_width"],
        )
        noise = torch.randn(shape, generator=generator, dtype=torch.float32)
        return noise.to(self.device, dtype=self.compute_dtype)

    def _write_run_manifest(self) -> None:
        manifest = {
            "mode": self.mode,
            "run_name": self.run_name,
            "training_regime": self.config.get(
                "training_regime", "autoregressive_rollout"
            ),
            "config": self.config,
            "parameter_counts": self.parameter_counts,
            "base_load_report": self.model.load_report,
            "special_token_count": len(self.model.vocabulary),
            "special_token_ids": self.model.vocabulary.as_dict(),
            "special_embedding_shape": list(self.model.special_embedding.weight.shape),
            "special_embedding_trainable": self.model.special_embedding.weight.requires_grad,
            "world_time_space_prior_shape": (
                list(self.model.world_time_space_prior.weight.shape)
                if hasattr(self.model, "world_time_space_prior")
                else None
            ),
            "gradient_checkpointing_blocks": self.model.num_gradient_checkpoint_blocks,
            "lora": self.model.lora_report,
            "lora_trainable_parameter_names": self.model.lora_parameter_names(),
            "lora_train_task_modules": bool(
                self.config.get("lora_train_task_modules", False)
            ),
            "warm_start": self.warm_start_report,
            "new_parameter_initialization": (
                "normal(std=0.02) embeddings; Xavier camera encoder; "
                "zero camera/world residual outputs and zero time-space world prior"
            ),
            "fixed_noise_sha256": self.fixed_noise_sha256,
            "camera_representation": CAMERA_REPRESENTATION,
            "action_vocabulary": self.config["action_vocabulary"],
            "preprocessing_config_hash": self.preprocessing_metadata["preprocessing_config_hash"],
        }
        with (self.run_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _history_window(self, history: list[DuplexTurn], current: DuplexTurn) -> list[DuplexTurn]:
        maximum = int(self.config["max_history_turns"])
        if maximum < 0:
            return [*history, current]
        if maximum == 0:
            return [current]
        return [*history[-maximum:], current]

    def _denoise_turn(
        self,
        turn_index: int,
        world_input: torch.Tensor,
        camera_input: torch.Tensor,
        history: list[DuplexTurn],
        target_state: torch.Tensor,
        target_camera: torch.Tensor,
    ) -> DenoisedTurn:
        initial_noise = self.fixed_noise[turn_index : turn_index + 1]
        noise_before = initial_noise.detach().clone()
        sample = initial_noise
        target_velocity = flow_target(target_state, initial_noise)
        flow_losses: list[torch.Tensor] = []
        final_camera: torch.Tensor | None = None
        sequence_lengths: list[int] = []
        for denoise_index in range(self.config["num_denoising_steps"]):
            sigma = self.sigmas[denoise_index]
            next_sigma = self.sigmas[denoise_index + 1]
            current = DuplexTurn(
                turn_index=turn_index,
                world_input=world_input,
                camera_input=camera_input,
                action_id=self.action_ids[turn_index : turn_index + 1],
                noise_input=sample,
            )
            visible_turns = self._history_window(history, current)
            timesteps = torch.zeros(
                (1, len(visible_turns)), device=self.device, dtype=torch.float32
            )
            timesteps[:, -1] = sigma * self.config["num_train_timesteps"]
            output = self.model(
                visible_turns,
                self.prompt_embedding,
                self.prompt_attention_mask,
                timesteps,
            )
            if not output.hidden_is_finite:
                raise FloatingPointError(f"Non-finite hidden state at turn {turn_index}, step {denoise_index}")
            if not torch.isfinite(output.flow).all() or not torch.isfinite(output.camera).all():
                raise FloatingPointError(f"Non-finite prediction at turn {turn_index}, step {denoise_index}")
            flow_losses.append(F.mse_loss(output.flow.float(), target_velocity.float()))
            sample = flow_step(sample, output.flow, sigma, next_sigma)
            final_camera = output.camera
            sequence_lengths.append(output.sequence_length)
        if not torch.equal(initial_noise, noise_before):
            raise AssertionError("Fixed initial noise was mutated during denoising")
        if not flow_losses or final_camera is None:
            raise AssertionError("Denoising loop failed to produce losses")
        mean_flow_loss = torch.stack(flow_losses).mean()
        state_loss = F.mse_loss(sample.float(), target_state.float())
        camera_losses = camera_loss(
            final_camera,
            target_camera,
            self.config["lambda_translation"],
            self.config["lambda_rotation"],
            self.config["lambda_intrinsics"],
        )
        total = (
            self.config["lambda_flow"] * mean_flow_loss
            + self.config["lambda_state"] * state_loss
            + self.config["lambda_camera"] * camera_losses.total
        )
        return DenoisedTurn(
            state=sample,
            camera=final_camera,
            flow_loss=mean_flow_loss,
            state_loss=state_loss,
            camera_losses=camera_losses,
            total_loss=total,
            sequence_lengths=sequence_lengths,
        )

    def forward_loss(self) -> StepOutput:
        num_turns = (
            1
            if self.mode == "single"
            else int(self.config.get("rollout_num_turns", self.config["num_micro_turns"]))
        )
        if not 1 <= num_turns <= self.config["num_micro_turns"]:
            raise ValueError("rollout_num_turns must be within the cached transition count")
        history: list[DuplexTurn] = []
        predictions: list[torch.Tensor] = []
        camera_predictions: list[torch.Tensor] = []
        results: list[DenoisedTurn] = []
        per_turn: list[dict[str, float]] = []
        all_sequence_lengths: list[int] = []
        for turn_index in range(num_turns):
            if turn_index == 0:
                world_input = torch.zeros_like(self.world_states[1:2]).unsqueeze(0)
                camera_input = self.cameras[0:1]
            else:
                world_input = predictions[-1]
                camera_input = camera_predictions[-1]
                if torch.is_grad_enabled() and (
                    world_input.grad_fn is None or camera_input.grad_fn is None
                ):
                    raise AssertionError("Rollout prediction was detached at a micro-turn boundary")
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
            predictions.append(result.state)
            camera_predictions.append(result.camera)
            results.append(result)
            history.append(
                DuplexTurn(
                    turn_index=turn_index,
                    world_input=world_input,
                    camera_input=camera_input,
                    action_id=self.action_ids[turn_index : turn_index + 1],
                    noise_input=self.fixed_noise[turn_index : turn_index + 1],
                    world_output=result.state,
                    camera_output=result.camera,
                )
            )
            all_sequence_lengths.extend(result.sequence_lengths)
            per_turn.append(
                {
                    "turn": float(turn_index),
                    "flow_loss": float(result.flow_loss.detach()),
                    "state_loss": float(result.state_loss.detach()),
                    "camera_loss": float(result.camera_losses.total.detach()),
                    "translation_loss": float(result.camera_losses.translation.detach()),
                    "rotation_loss": float(result.camera_losses.rotation.detach()),
                    "intrinsics_loss": float(result.camera_losses.intrinsics.detach()),
                }
            )

        total_loss = torch.stack([result.total_loss for result in results]).mean()
        flow_loss_value = torch.stack([result.flow_loss for result in results]).mean()
        state_loss_value = torch.stack([result.state_loss for result in results]).mean()
        camera_loss_value = torch.stack([result.camera_losses.total for result in results]).mean()
        translation = torch.stack([result.camera_losses.translation for result in results]).mean()
        rotation = torch.stack([result.camera_losses.rotation for result in results]).mean()
        intrinsics = torch.stack([result.camera_losses.intrinsics for result in results]).mean()

        future_gradient_norm: float | None = None
        early_turn_prediction_probe: torch.Tensor | None = None
        early_turn_local_gradient: torch.Tensor | None = None
        if self.mode == "rollout" and self.global_step == 0 and num_turns > 1:
            # Retain one non-leaf gradient and recover the future-turn
            # contribution after the ordinary total-loss backward. This avoids
            # traversing the complete 19-turn graph twice. The only direct
            # turn-0 objective downstream of its final state tensor is L_state;
            # flow predictions precede the final Euler update and camera uses a
            # separate tensor.
            early_turn_prediction_probe = predictions[0]
            early_turn_prediction_probe.retain_grad()
            local_objective = (
                self.config["lambda_state"] * results[0].state_loss / num_turns
            )
            early_turn_local_gradient = torch.autograd.grad(
                local_objective,
                early_turn_prediction_probe,
                retain_graph=True,
                allow_unused=False,
            )[0].detach()

        return StepOutput(
            total_loss=total_loss,
            flow_loss=flow_loss_value,
            state_loss=state_loss_value,
            camera_loss=camera_loss_value,
            translation_loss=translation,
            rotation_loss=rotation,
            intrinsics_loss=intrinsics,
            predictions=predictions,
            camera_predictions=camera_predictions,
            per_turn=per_turn,
            sequence_lengths=all_sequence_lengths,
            early_turn_future_gradient_norm=future_gradient_norm,
            early_turn_prediction_probe=early_turn_prediction_probe,
            early_turn_local_gradient=early_turn_local_gradient,
        )

    def _parameter_norm(self) -> float:
        accumulator = torch.zeros((), device=self.device)
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                accumulator += parameter.detach().float().pow(2).sum()
        return float(accumulator.sqrt())

    def train_step(self) -> dict[str, Any]:
        # Release the prior step's retained prediction graph before building the
        # next one, then enable within-step reuse of static history encodings.
        self.last_step_output = None
        self.model.clear_step_cache()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        with torch.autocast(
            device_type="cuda",
            dtype=self.compute_dtype,
            enabled=self.compute_dtype == torch.bfloat16,
        ):
            output = self.forward_loss()
        if not torch.isfinite(output.total_loss):
            raise FloatingPointError(f"Non-finite total loss at step {self.global_step}")
        output.total_loss.backward()
        if output.early_turn_prediction_probe is not None:
            probe_gradient = output.early_turn_prediction_probe.grad
            local_gradient = output.early_turn_local_gradient
            if probe_gradient is None or local_gradient is None:
                raise AssertionError("Early-turn gradient probe was not populated")
            future_gradient = probe_gradient.detach() - local_gradient
            if not torch.isfinite(future_gradient).all():
                raise FloatingPointError("Early-turn future gradient is non-finite")
            output.early_turn_future_gradient_norm = float(future_gradient.float().norm())
            if output.early_turn_future_gradient_norm == 0.0:
                raise AssertionError(
                    "Turn-0 prediction received zero gradient from future-turn losses"
                )
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Non-finite gradient: {name}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            self.config["max_grad_norm"],
        )
        parameter_norm = self._parameter_norm()
        self.optimizer.step()
        self.lr_scheduler.step()
        torch.cuda.synchronize()
        self.global_step += 1
        self.epoch += 1
        elapsed = time.perf_counter() - started
        metrics = {
            "step": self.global_step,
            "epoch": self.epoch,
            "total_loss": float(output.total_loss.detach()),
            "flow_loss": float(output.flow_loss.detach()),
            "state_loss": float(output.state_loss.detach()),
            "camera_loss": float(output.camera_loss.detach()),
            "translation_loss": float(output.translation_loss.detach()),
            "rotation_loss": float(output.rotation_loss.detach()),
            "intrinsics_loss": float(output.intrinsics_loss.detach()),
            "gradient_norm": float(gradient_norm),
            "parameter_norm": parameter_norm,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "world_head_learning_rate": next(
                (
                    group["lr"]
                    for group in self.optimizer.param_groups
                    if group.get("group_name") == "world_head"
                ),
                self.optimizer.param_groups[0]["lr"],
            ),
            "world_prior_learning_rate": next(
                (
                    group["lr"]
                    for group in self.optimizer.param_groups
                    if group.get("group_name") == "world_prior"
                ),
                self.optimizer.param_groups[0]["lr"],
            ),
            "lora_learning_rate": next(
                (
                    group["lr"]
                    for group in self.optimizer.param_groups
                    if group.get("group_name") == "lora"
                ),
                self.optimizer.param_groups[0]["lr"],
            ),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
            "elapsed_seconds": elapsed,
            "latent_prediction_mse": float(output.state_loss.detach()),
            "early_turn_future_gradient_norm": output.early_turn_future_gradient_norm,
            "min_sequence_length": min(output.sequence_lengths),
            "max_sequence_length": max(output.sequence_lengths),
            "per_turn": output.per_turn,
        }
        self.loss_history.append(metrics)
        self.last_step_output = output
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(f"[train] {json.dumps(metrics, sort_keys=True)}", flush=True)
        return metrics

    def _model_state_for_checkpoint(self) -> tuple[str, dict[str, torch.Tensor], list[str]]:
        full_state = self.model.state_dict()
        if self.config["train_backbone"]:
            keys = list(full_state)
            state_format = "full"
        elif self.config.get("lora_enabled", False):
            # LoRA-only training freezes the warm-started Full-Duplex task
            # modules, but the checkpoint must still contain them. Otherwise a
            # fresh strict-base reload would recreate random task embeddings and
            # heads and silently produce a different model.
            keys = self.model.adapter_state_names()
            state_format = "full_duplex_lora_delta_over_strict_base"
        else:
            trainable_names = {
                name for name, parameter in self.model.named_parameters() if parameter.requires_grad
            }
            keys = sorted(trainable_names)
            state_format = "trainable_delta_over_strict_base"
        state = {key: full_state[key].detach().cpu() for key in keys}
        return state_format, state, keys

    def checkpoint_payload(self) -> dict[str, Any]:
        state_format, model_state, model_keys = self._model_state_for_checkpoint()
        return {
            "model": model_state,
            "model_state_format": state_format,
            "model_keys": model_keys,
            "trainable_model_keys": sorted(
                name for name, parameter in self.model.named_parameters() if parameter.requires_grad
            ),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "loss_history": self.loss_history,
            "random_seed": self.config["seed"],
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "pytorch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "model_config": {
                "model_dim": self.config["model_dim"],
                "num_backbone_blocks": self.config["num_backbone_blocks"],
                "spatial_token_stride": self.config["spatial_token_stride"],
                "use_prope": self.config["use_prope"],
                "world_residual_head": self.config.get("world_residual_head", False),
                "world_time_space_prior": self.config.get("world_time_space_prior", False),
                "train_base_world_head": self.config.get("train_base_world_head", False),
                "lora_enabled": self.config.get("lora_enabled", False),
                "lora_last_blocks": self.config.get("lora_last_blocks"),
                "lora_rank": self.config.get("lora_rank"),
                "lora_alpha": self.config.get("lora_alpha"),
                "lora_dropout": self.config.get("lora_dropout"),
                "lora_targets": self.config.get("lora_targets"),
                "lora_train_task_modules": self.config.get(
                    "lora_train_task_modules", False
                ),
                "training_regime": self.config.get(
                    "training_regime", "autoregressive_rollout"
                ),
                "gradient_checkpointing": self.config["gradient_checkpointing"],
                "gradient_checkpointing_blocks": self.model.num_gradient_checkpoint_blocks,
            },
            "training_config": self.config,
            "token_vocabulary": self.model.vocabulary.as_dict(),
            "token_ids": self.model.vocabulary.as_dict(),
            "camera_representation": CAMERA_REPRESENTATION,
            "action_vocabulary": self.config["action_vocabulary"],
            "preprocessing_metadata_hash": self.preprocessing_metadata[
                "preprocessing_config_hash"
            ],
            "base_checkpoint_identity": self.preprocessing_metadata["identities"][
                "base_checkpoint"
            ],
            "base_load_report": self.model.load_report,
            "fixed_noise_seed": self.config["fixed_noise_seed"],
            "fixed_noise_sha256": self.fixed_noise_sha256,
            "mode": self.mode,
            "run_name": self.run_name,
            "warm_start": self.warm_start_report,
        }

    def save_checkpoint(self, metrics: dict[str, Any], *, named: bool, best: bool) -> Path | None:
        payload = self.checkpoint_payload()
        result: Path | None = None
        if named:
            name = (
                f"step_{self.global_step:06d}_total_{metrics['total_loss']:.6f}_"
                f"state_{metrics['state_loss']:.6f}_camera_{metrics['camera_loss']:.6f}.pt"
            )
            result = self.checkpoint_dir / name
            _atomic_torch_save(payload, result)
            _atomic_symlink(result, self.checkpoint_dir / "latest.pt")
            print(f"[checkpoint] saved {result}", flush=True)
        if best:
            best_path = self.checkpoint_dir / "best.pt"
            _atomic_torch_save(payload, best_path)
            print(f"[checkpoint] updated train-set best {best_path}", flush=True)
            if result is None:
                result = best_path
        return result

    def load_warm_start(self, path: str | Path) -> dict[str, Any]:
        """Load a prior Full-Duplex task delta while resetting training state.

        This differs intentionally from resume: architecture/runtime controls
        may change, optimizer moments and global step are not inherited, and
        every current LoRA matrix must still have its zero initial residual.
        The later LoRA checkpoints save both this frozen task delta and LoRA,
        so they remain independently reloadable over the strict base model.
        """

        if not self.config.get("lora_enabled", False):
            raise RuntimeError("Warm-start is currently reserved for explicit LoRA training")
        checkpoint_path = Path(path).resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_state_format") != "trainable_delta_over_strict_base":
            raise RuntimeError(
                "LoRA warm-start requires a pre-LoRA trainable_delta_over_strict_base checkpoint"
            )
        if checkpoint["preprocessing_metadata_hash"] != self.preprocessing_metadata[
            "preprocessing_config_hash"
        ]:
            raise RuntimeError("Warm-start preprocessing metadata hash mismatch")
        if checkpoint["fixed_noise_sha256"] != self.fixed_noise_sha256:
            raise RuntimeError("Warm-start fixed-noise identity mismatch")
        if checkpoint["base_checkpoint_identity"] != self.preprocessing_metadata[
            "identities"
        ]["base_checkpoint"]:
            raise RuntimeError("Warm-start strict-base checkpoint identity mismatch")

        state = checkpoint["model"]
        source_keys = set(checkpoint["model_keys"])
        if set(state) != source_keys:
            raise RuntimeError("Warm-start source delta key manifest mismatch")
        required_task_keys = set(self.model.new_parameter_names())
        if source_keys != required_task_keys:
            raise RuntimeError(
                "Warm-start task-module mismatch: "
                f"missing_in_source={sorted(required_task_keys - source_keys)}, "
                f"unexpected_in_source={sorted(source_keys - required_task_keys)}"
            )
        current_state = self.model.state_dict()
        for key, value in state.items():
            if key not in current_state:
                raise RuntimeError(f"Warm-start key absent from current model: {key}")
            if value.shape != current_state[key].shape:
                raise RuntimeError(
                    f"Warm-start shape mismatch for {key}: {value.shape} vs "
                    f"{current_state[key].shape}"
                )
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Warm-start tensor is non-finite: {key}")

        incompatible = self.model.load_state_dict(state, strict=False)
        missing, unexpected = list(incompatible.missing_keys), list(incompatible.unexpected_keys)
        allowed_missing = set(current_state) - source_keys
        if set(missing) != allowed_missing or unexpected:
            raise RuntimeError(
                f"Warm-start load mismatch missing={missing}, unexpected={unexpected}"
            )
        assert_zero_lora_residual(self.model)
        previous_config = checkpoint["training_config"]
        report = {
            "source_checkpoint": str(checkpoint_path),
            "source_global_step": int(checkpoint["global_step"]),
            "source_best_step": int(checkpoint["best_step"]),
            "source_best_loss": float(checkpoint["best_loss"]),
            "source_model_state_format": checkpoint["model_state_format"],
            "loaded_task_keys": sorted(source_keys),
            "loaded_task_elements": int(sum(value.numel() for value in state.values())),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "lora_zero_residual_verified": True,
            "architecture_changes": {
                "num_backbone_blocks": {
                    "source": previous_config["num_backbone_blocks"],
                    "current": self.config["num_backbone_blocks"],
                },
                "spatial_token_stride": {
                    "source": previous_config["spatial_token_stride"],
                    "current": self.config["spatial_token_stride"],
                },
                "lora_enabled": {"source": False, "current": True},
            },
            "optimizer_state_inherited": False,
            "global_step_reset_to_zero": True,
        }
        self.warm_start_report = report
        self._write_run_manifest()
        report_path = self.run_dir / "warm_start_report.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(
            f"[warm start] loaded task delta keys({len(source_keys)}) from "
            f"{checkpoint_path}; missing({len(missing)})={missing}; unexpected={unexpected}",
            flush=True,
        )
        return report

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        override_resume_learning_rate: bool = False,
    ) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        resume_keys = (
            "training_regime",
            "teacher_forced_world_inputs",
            "teacher_force_camera",
            "sequential_turn_backward",
            "teacher_forcing_ratio",
            "detach_between_turns",
            "base_checkpoint",
            "cache_path",
            "num_micro_turns",
            "num_denoising_steps",
            "model_dim",
            "num_backbone_blocks",
            "spatial_token_stride",
            "max_history_turns",
            "attention_padding_strategy",
            "attention_pad_to_turns",
            "train_backbone",
            "train_base_world_head",
            "world_residual_head",
            "world_time_space_prior",
            "lora_enabled",
            "lora_last_blocks",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_targets",
            "lora_train_task_modules",
            "use_prope",
            "learning_rate",
            "world_head_learning_rate_multiplier",
            "world_prior_learning_rate_multiplier",
            "lora_learning_rate_multiplier",
            "weight_decay",
            "lambda_flow",
            "lambda_state",
            "lambda_camera",
            "lambda_translation",
            "lambda_rotation",
            "lambda_intrinsics",
            "seed",
            "fixed_noise_seed",
        )
        previous_config = checkpoint["training_config"]
        resume_defaults = {
            "training_regime": "autoregressive_rollout",
            "teacher_forced_world_inputs": False,
            "teacher_force_camera": False,
            "sequential_turn_backward": False,
            "teacher_forcing_ratio": 0.0,
            "detach_between_turns": False,
            "world_residual_head": False,
            "world_time_space_prior": False,
            "train_base_world_head": False,
            "world_head_learning_rate_multiplier": 1.0,
            "world_prior_learning_rate_multiplier": 1.0,
            "lora_enabled": False,
            "lora_last_blocks": 4,
            "lora_rank": 8,
            "lora_alpha": 8.0,
            "lora_dropout": 0.0,
            "lora_targets": None,
            "lora_train_task_modules": False,
            "lora_learning_rate_multiplier": 1.0,
        }
        mismatches = {}
        for key in resume_keys:
            if key.startswith("lora_") and not previous_config.get(
                "lora_enabled", False
            ) and not self.config.get("lora_enabled", False):
                continue
            default = resume_defaults.get(key)
            previous_value = previous_config.get(key, default)
            current_value = self.config.get(key, default)
            if previous_value != current_value:
                mismatches[key] = {
                    "checkpoint": previous_value,
                    "current": current_value,
                }
        allowed_lr_mismatches = {}
        for key in (
            "learning_rate",
            "world_head_learning_rate_multiplier",
            "world_prior_learning_rate_multiplier",
            "lora_learning_rate_multiplier",
        ):
            mismatch = mismatches.pop(key, None)
            if mismatch is not None:
                allowed_lr_mismatches[key] = mismatch
        if allowed_lr_mismatches and not override_resume_learning_rate:
            mismatches.update(allowed_lr_mismatches)
        if mismatches:
            raise RuntimeError(f"Resume configuration mismatch: {mismatches}")
        if checkpoint["fixed_noise_sha256"] != self.fixed_noise_sha256:
            raise RuntimeError("Resume fixed-noise identity mismatch")
        state = checkpoint["model"]
        state_format = checkpoint["model_state_format"]
        if state_format == "full":
            incompatible = self.model.load_state_dict(state, strict=True)
            missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
        elif state_format == "trainable_delta_over_strict_base":
            expected = set(checkpoint["model_keys"])
            if set(state) != expected:
                raise RuntimeError("Checkpoint delta key manifest mismatch")
            current_trainable = {
                name for name, parameter in self.model.named_parameters() if parameter.requires_grad
            }
            if expected != current_trainable:
                raise RuntimeError(
                    "Checkpoint/current trainable-module mismatch: "
                    f"missing_in_checkpoint={sorted(current_trainable - expected)}, "
                    f"unexpected_in_checkpoint={sorted(expected - current_trainable)}"
                )
            incompatible = self.model.load_state_dict(state, strict=False)
            missing, unexpected = list(incompatible.missing_keys), list(incompatible.unexpected_keys)
            allowed_missing = set(self.model.state_dict()) - expected
            if set(missing) != allowed_missing or unexpected:
                raise RuntimeError(
                    f"Delta reload mismatch missing={missing}, unexpected={unexpected}"
                )
        elif state_format == "full_duplex_lora_delta_over_strict_base":
            expected = set(checkpoint["model_keys"])
            if set(state) != expected:
                raise RuntimeError("LoRA checkpoint adapter-state key manifest mismatch")
            current_adapter = set(self.model.adapter_state_names())
            if expected != current_adapter:
                raise RuntimeError(
                    "LoRA checkpoint/current adapter mismatch: "
                    f"missing_in_checkpoint={sorted(current_adapter - expected)}, "
                    f"unexpected_in_checkpoint={sorted(expected - current_adapter)}"
                )
            saved_trainable = set(checkpoint["trainable_model_keys"])
            current_trainable = {
                name for name, parameter in self.model.named_parameters() if parameter.requires_grad
            }
            if saved_trainable != current_trainable:
                raise RuntimeError(
                    "LoRA checkpoint/current trainable mismatch: "
                    f"missing_in_checkpoint={sorted(current_trainable - saved_trainable)}, "
                    f"unexpected_in_checkpoint={sorted(saved_trainable - current_trainable)}"
                )
            incompatible = self.model.load_state_dict(state, strict=False)
            missing, unexpected = list(incompatible.missing_keys), list(incompatible.unexpected_keys)
            allowed_missing = set(self.model.state_dict()) - expected
            if set(missing) != allowed_missing or unexpected:
                raise RuntimeError(
                    f"LoRA delta reload mismatch missing={missing}, unexpected={unexpected}"
                )
        else:
            raise ValueError(f"Unknown model_state_format {state_format}")
        print(
            f"[checkpoint reload] format={state_format} missing({len(missing)})={missing} "
            f"unexpected({len(unexpected)})={unexpected}",
            flush=True,
        )
        saved_optimizer = checkpoint["optimizer"]
        optimizer_group_migration = len(saved_optimizer["param_groups"]) != len(
            self.optimizer.param_groups
        )
        if optimizer_group_migration:
            saved_parameter_ids = [
                parameter_id
                for group in saved_optimizer["param_groups"]
                for parameter_id in group["params"]
            ]
            migrated_optimizer = self.optimizer.state_dict()
            current_parameter_ids = [
                parameter_id
                for group in migrated_optimizer["param_groups"]
                for parameter_id in group["params"]
            ]
            if saved_parameter_ids != current_parameter_ids:
                raise RuntimeError(
                    "Cannot migrate optimizer groups because parameter order changed"
                )
            migrated_optimizer["state"] = saved_optimizer["state"]
            self.optimizer.load_state_dict(migrated_optimizer)
            saved_scheduler = checkpoint["lr_scheduler"]
            self.lr_scheduler.last_epoch = saved_scheduler["last_epoch"]
            self.lr_scheduler._step_count = saved_scheduler["_step_count"]
            print(
                "[checkpoint reload] migrated optimizer state from "
                f"{len(saved_optimizer['param_groups'])} to "
                f"{len(self.optimizer.param_groups)} LR groups without dropping moments",
                flush=True,
            )
        else:
            self.optimizer.load_state_dict(saved_optimizer)
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if override_resume_learning_rate:
            resumed_lr = float(self.config["learning_rate"])
            world_head_multiplier = float(
                self.config.get("world_head_learning_rate_multiplier", 1.0)
            )
            world_prior_multiplier = float(
                self.config.get("world_prior_learning_rate_multiplier", 1.0)
            )
            lora_multiplier = float(
                self.config.get("lora_learning_rate_multiplier", 1.0)
            )
            for parameter_group in self.optimizer.param_groups:
                group_lr = resumed_lr
                if parameter_group.get("group_name") == "world_head":
                    group_lr *= world_head_multiplier
                elif parameter_group.get("group_name") == "world_prior":
                    group_lr *= world_prior_multiplier
                elif parameter_group.get("group_name") == "lora":
                    group_lr *= lora_multiplier
                parameter_group["lr"] = group_lr
                parameter_group["initial_lr"] = group_lr
            self.lr_scheduler.base_lrs = [
                group["lr"] for group in self.optimizer.param_groups
            ]
            self.lr_scheduler._last_lr = list(self.lr_scheduler.base_lrs)
            print(
                "[checkpoint reload] explicitly overrode optimizer learning rates: "
                + ", ".join(
                    f"{group.get('group_name', index)}={group['lr']}"
                    for index, group in enumerate(self.optimizer.param_groups)
                ),
                flush=True,
            )
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.best_step = checkpoint["best_step"]
        self.loss_history = checkpoint["loss_history"]
        self.warm_start_report = checkpoint.get("warm_start")
        self._write_run_manifest()
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["pytorch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    def _synchronize_resume_artifacts(self, resume: str | Path) -> None:
        """Seed a new run directory without duplicating historical metrics."""
        existing: list[dict[str, Any]] = []
        if self.log_path.exists():
            existing = [
                json.loads(line)
                for line in self.log_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
        if existing != self.loss_history[: len(existing)]:
            raise RuntimeError("Existing metrics.jsonl is not a prefix of resumed loss history")
        if len(existing) < len(self.loss_history):
            with self.log_path.open("a", encoding="utf-8") as handle:
                for metric in self.loss_history[len(existing) :]:
                    handle.write(json.dumps(metric, sort_keys=True) + "\n")

        latest = self.checkpoint_dir / "latest.pt"
        best = self.checkpoint_dir / "best.pt"
        if not latest.exists():
            last_metric = self.loss_history[-1]
            self.save_checkpoint(
                last_metric,
                named=True,
                best=self.global_step == self.best_step,
            )
        if not best.exists():
            source_best = Path(resume).resolve().parent / "best.pt"
            if not source_best.exists():
                raise RuntimeError("Resume source has no best.pt to seed the new run")
            best_payload = torch.load(source_best, map_location="cpu", weights_only=False)
            _atomic_torch_save(best_payload, best)
            print(f"[checkpoint] copied prior train-set best {source_best} -> {best}")

    @torch.inference_mode()
    def _reload_probe(self, model: FullDuplexWanModel) -> tuple[torch.Tensor, torch.Tensor]:
        model.clear_step_cache()
        model.eval()
        target = self.world_states[1:2].unsqueeze(0)
        turn = DuplexTurn(
            turn_index=0,
            world_input=torch.zeros_like(target),
            camera_input=self.cameras[0:1],
            action_id=self.action_ids[0:1],
            noise_input=self.fixed_noise[0:1],
        )
        timestep = torch.full((1, 1), 500.0, device=self.device)
        with torch.autocast(
            "cuda", dtype=self.compute_dtype, enabled=self.compute_dtype == torch.bfloat16
        ):
            output = model(
                [turn], self.prompt_embedding, self.prompt_attention_mask, timestep
            )
        return output.flow.detach().cpu(), output.camera.detach().cpu()

    def checkpoint_reload_test(self, checkpoint_path: Path) -> dict[str, float]:
        before_flow, before_camera = self._reload_probe(self.model)
        fresh = FullDuplexWanModel.from_checkpoint(self.config)
        fresh.configure_trainable_parameters(
            self.config["train_backbone"],
            self.config.get("train_base_world_head", False),
        )
        fresh = fresh.to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model"]
        state_format = checkpoint["model_state_format"]
        if state_format == "full":
            incompatible = fresh.load_state_dict(state, strict=True)
            missing, unexpected = list(incompatible.missing_keys), list(incompatible.unexpected_keys)
        elif state_format == "trainable_delta_over_strict_base":
            expected = set(checkpoint["model_keys"])
            current_trainable = {
                name for name, parameter in fresh.named_parameters() if parameter.requires_grad
            }
            if expected != current_trainable:
                raise RuntimeError("Fresh delta reload trainable-module manifest mismatch")
            incompatible = fresh.load_state_dict(state, strict=False)
            missing, unexpected = list(incompatible.missing_keys), list(incompatible.unexpected_keys)
            if set(missing) != set(fresh.state_dict()) - expected or unexpected:
                raise RuntimeError("Fresh delta reload failed strict expected-key validation")
        elif state_format == "full_duplex_lora_delta_over_strict_base":
            expected = set(checkpoint["model_keys"])
            if expected != set(fresh.adapter_state_names()):
                raise RuntimeError("Fresh LoRA reload adapter-state manifest mismatch")
            current_trainable = {
                name for name, parameter in fresh.named_parameters() if parameter.requires_grad
            }
            if current_trainable != set(checkpoint["trainable_model_keys"]):
                raise RuntimeError("Fresh LoRA reload trainable-module manifest mismatch")
            incompatible = fresh.load_state_dict(state, strict=False)
            missing, unexpected = list(incompatible.missing_keys), list(incompatible.unexpected_keys)
            if set(missing) != set(fresh.state_dict()) - expected or unexpected:
                raise RuntimeError("Fresh LoRA reload failed strict expected-key validation")
        else:
            raise ValueError(f"Unknown model_state_format {state_format}")
        print(
            f"[reload test] missing({len(missing)})={missing} unexpected({len(unexpected)})={unexpected}"
        )
        after_flow, after_camera = self._reload_probe(fresh)
        flow_error = float((before_flow - after_flow).abs().max())
        camera_error = float((before_camera - after_camera).abs().max())
        if flow_error != 0.0 or camera_error != 0.0:
            raise AssertionError(
                f"Checkpoint reload changed output: flow={flow_error}, camera={camera_error}"
            )
        del fresh, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        report = {"flow_max_abs_error": flow_error, "camera_max_abs_error": camera_error}
        with (self.run_dir / "reload_test.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return report

    def save_predictions(self) -> Path:
        if self.last_step_output is None:
            raise RuntimeError("No predictions to save")
        path = self.run_dir / "last_predictions.pt"
        payload = {
            "states": torch.cat(
                [prediction.detach().cpu() for prediction in self.last_step_output.predictions], dim=1
            ),
            "cameras": torch.cat(
                [prediction.detach().cpu() for prediction in self.last_step_output.camera_predictions], dim=0
            ),
            "target_states": self.world_states[
                1 : len(self.last_step_output.predictions) + 1
            ].detach().cpu(),
            "target_cameras": self.cameras[
                1 : len(self.last_step_output.camera_predictions) + 1
            ].detach().cpu(),
        }
        _atomic_torch_save(payload, path)
        return path

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
            self.load_warm_start(warm_start)
        if resume is not None:
            self.load_checkpoint(
                resume,
                override_resume_learning_rate=override_resume_learning_rate,
            )
            self._synchronize_resume_artifacts(resume)
        initial_loss = self.loss_history[0]["total_loss"] if self.loss_history else None
        final_checkpoint: Path | None = None
        try:
            while self.global_step < max_steps:
                metrics = self.train_step()
                if initial_loss is None:
                    initial_loss = metrics["total_loss"]
                improved = metrics["total_loss"] < self.best_loss
                if improved:
                    self.best_loss = metrics["total_loss"]
                    self.best_step = self.global_step
                milestone = self.global_step in (1, 10, 100, 500, 1000)
                named = (
                    milestone
                    or self.global_step % self.config["save_every"] == 0
                    or self.global_step == max_steps
                )
                # Delta checkpoints are small enough to preserve every new train best.
                save_best = improved
                if named or save_best:
                    final_checkpoint = self.save_checkpoint(metrics, named=named, best=save_best)
        except BaseException as error:
            with (self.run_dir / "exception.log").open("a", encoding="utf-8") as handle:
                handle.write(f"step={self.global_step} {type(error).__name__}: {error}\n")
            raise
        predictions_path = self.save_predictions()
        latest = self.checkpoint_dir / "latest.pt"
        if not latest.exists():
            raise RuntimeError("Training ended without latest.pt")
        reload_report = self.checkpoint_reload_test(latest.resolve())
        losses = [entry["total_loss"] for entry in self.loss_history]
        summary = {
            "mode": self.mode,
            "run_name": self.run_name,
            "steps": self.global_step,
            "initial_loss": initial_loss if initial_loss is not None else losses[0],
            "final_loss": losses[-1],
            "minimum_loss": min(losses),
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "best_checkpoint": str((self.checkpoint_dir / "best.pt").resolve()),
            "latest_checkpoint": str(latest.resolve()),
            "predictions": str(predictions_path),
            "reload_test": reload_report,
            "fixed_noise_sha256": self.fixed_noise_sha256,
            "train_set_overfit_metric": True,
        }
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"[training summary] {json.dumps(summary, sort_keys=True)}")
        return summary
