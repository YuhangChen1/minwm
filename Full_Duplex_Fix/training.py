from __future__ import annotations

import json
import math
import os
import random
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .checkpoint import load_strict_generator
from .checkpoint_io import replace_with_link
from .config import resolved_config_for_json
from .data import CachedSample, load_cached_sample, tensor_sha256
from .debug.tracer import (
    DebugTracer,
    active_debug_tracer,
    capture_parameter_values,
    debug_scope,
    debug_timer,
    parameter_updates,
    trace_event,
    trace_tensor,
)
from .flow import (
    FlowTrainingBatch,
    flow_matching_losses,
    flow_to_clean,
    sample_flow_training_batch,
)
from .model import InterleavedWanAdapter
from .wandb_tracking import WandbTracker


CHECKPOINT_VERSION = "full_duplex_fix_checkpoint_v1"


RESUME_CONTRACT_KEYS = (
    "generator_checkpoint_stage",
    "expected_base_checkpoint_sha256",
    "layout_version",
    "mask_version",
    "camera_protocol",
    "rope_protocol",
    "num_states",
    "num_noisy_spans",
    "num_clean_spans",
    "num_total_spans",
    "tokens_per_span",
    "sequence_length",
    "num_frame_per_block",
    "num_train_timesteps",
    "timestep_shift",
    "learning_rate",
    "weight_decay",
    "beta1",
    "beta2",
    "lr_scheduler",
    "mixed_precision",
    "gradient_checkpointing",
    "fixed_evaluation_seed",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_value, sort_keys=True) + "\n")


def git_identity(project_root: str | Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ("git", *args),
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "status_porcelain": run("status", "--short"),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def latent_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    prediction_float = prediction.float()
    target_float = target.float()
    delta = prediction_float - target_float
    dims = (2, 3, 4)
    per_state_mse = delta.square().mean(dim=dims).mean(dim=0)
    flat_prediction = prediction_float.flatten(2)
    flat_target = target_float.flatten(2)
    per_example_cosine = torch.nn.functional.cosine_similarity(
        flat_prediction, flat_target, dim=-1
    )
    per_state_cosine = per_example_cosine.mean(dim=0)
    zero_baseline = target_float.square().mean()
    return {
        "latent_mse": float(delta.square().mean()),
        "latent_cosine": float(per_example_cosine.mean()),
        "per_state_mse": per_state_mse.detach().cpu().tolist(),
        "per_state_cosine": per_state_cosine.detach().cpu().tolist(),
        "prediction_mean": float(prediction_float.mean()),
        "prediction_std": float(prediction_float.std()),
        "target_mean": float(target_float.mean()),
        "target_std": float(target_float.std()),
        "zero_latent_baseline_mse": float(zero_baseline),
        "beats_zero_baseline": bool(delta.square().mean() < zero_baseline),
        "bootstrap_mse": float(per_state_mse[0]),
        "transition_mse": float(per_state_mse[1:].mean()),
        "bootstrap_cosine": float(per_state_cosine[0]),
        "transition_cosine": float(per_state_cosine[1:].mean()),
    }


class OverfitTrainer:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        device: torch.device | str,
        wandb_overrides: Mapping[str, Any] | None = None,
        debug_mode: bool = False,
        debug_tracer: DebugTracer | None = None,
    ) -> None:
        self.debug_mode = bool(debug_mode)
        self.debug_tracer = debug_tracer
        if self.debug_mode and self.debug_tracer is None:
            raise ValueError("debug_mode requires a DebugTracer")
        self.config = dict(config)
        tracking_overrides = dict(wandb_overrides or {})
        extra_tags = tracking_overrides.pop("wandb_tags", None)
        self.config.update(tracking_overrides)
        if extra_tags is not None:
            self.config["wandb_tags"] = list(
                dict.fromkeys([*self.config.get("wandb_tags", ()), *extra_tags])
            )
        config = self.config
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The real Wan overfit trainer requires CUDA")
        torch.cuda.set_device(self.device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        set_seed(int(config["seed"]))
        if self.debug_mode:
            trace_event(
                "session",
                "session.training_configuration",
                details={
                    "device": str(self.device),
                    "layout_version": config["layout_version"],
                    "num_total_spans": config["num_total_spans"],
                    "sequence_length": config["sequence_length"],
                    "mixed_precision": config["mixed_precision"],
                    "gradient_checkpointing": config["gradient_checkpointing"],
                    "base_checkpoint": config["base_checkpoint"],
                },
            )

        self.compute_dtype = (
            torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
        )
        with debug_timer() as timing:
            self.sample: CachedSample = load_cached_sample(config["cache_path"])
        if self.debug_mode:
            trace_event(
                "data",
                "session.load_cached_sample",
                tensors={
                    "world_latents": self.sample.world_latents,
                    "prompt_embedding": self.sample.prompt_embedding,
                    "viewmats": self.sample.viewmats,
                    "Ks": self.sample.Ks,
                },
                details={
                    **timing,
                    "cache_path": config["cache_path"],
                    "source_video": self.sample.metadata.get("video"),
                    "preencode_runtime_seconds": self.sample.metadata.get("runtime_seconds"),
                    "vae": "offline frozen encoding; not loaded in the five-step training graph",
                    "t5": "offline frozen encoding; not loaded in the five-step training graph",
                },
            )
        with debug_timer() as timing:
            generator, self.base_checkpoint_audit = load_strict_generator(
                config, device=self.device, dtype=torch.float32
            )
        if self.debug_mode:
            trace_event(
                "checkpoint",
                "session.strict_load_ar_checkpoint",
                details={
                    **timing,
                    "strict_load": self.base_checkpoint_audit["strict_load"],
                    "parameter_count": self.base_checkpoint_audit["parameter_count"],
                    "prope_layers": self.base_checkpoint_audit["prope_layers"],
                },
            )
        with debug_timer() as timing:
            self.model = InterleavedWanAdapter(
                generator,
                gradient_checkpointing=bool(config["gradient_checkpointing"]),
            ).to(self.device)
        if self.debug_mode:
            trace_event(
                "model",
                "session.build_interleaved_adapter",
                details={**timing, **self.model.parameter_manifest()},
            )
        with debug_timer() as timing:
            self.optimizer = torch.optim.AdamW(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                lr=float(config["learning_rate"]),
                betas=(float(config["beta1"]), float(config["beta2"])),
                weight_decay=float(config["weight_decay"]),
            )
        if self.debug_mode:
            trace_event(
                "optimizer",
                "session.build_adamw",
                details={
                    **timing,
                    "learning_rate": float(config["learning_rate"]),
                    "betas": [float(config["beta1"]), float(config["beta2"])],
                    "weight_decay": float(config["weight_decay"]),
                },
            )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda _: 1.0,
        )
        self.global_step = 0
        self.best_metric = math.inf
        self.best_step = -1
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb = WandbTracker(config, self.output_dir)
        self.restored_wandb_identity: dict[str, Any] | None = None
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.eval_path = self.output_dir / "evaluation.jsonl"
        self.raw_log_path = self.output_dir / "raw_log.jsonl"
        self.fixed_batch = self._make_fixed_batch()
        self._write_run_manifest()

    def _resume_contract(self) -> dict[str, Any]:
        return {key: self.config[key] for key in RESUME_CONTRACT_KEYS}

    def _sample_tensors(self) -> dict[str, torch.Tensor]:
        return self.sample.batched(self.device, self.compute_dtype)

    def _make_fixed_batch(self) -> FlowTrainingBatch:
        tensors = self._sample_tensors()
        generator = torch.Generator(device=self.device).manual_seed(
            int(self.config["fixed_evaluation_seed"])
        )
        return sample_flow_training_batch(
            tensors["world_latents"], self.model.scheduler, generator=generator
        )

    def _write_run_manifest(self) -> None:
        manifest = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "resolved_config": resolved_config_for_json(self.config),
            "base_checkpoint": self.base_checkpoint_audit,
            "parameters": self.model.parameter_manifest(),
            "layout": {
                "name": self.model.layout.name,
                "labels": self.model.layout.labels(),
                "spans": [asdict(span) for span in self.model.layout.spans],
                "sequence_length": self.model.layout.sequence_length,
            },
            "cache_preprocessing_hash": self.sample.metadata["preprocessing_hash"],
            "cache_tensor_sha256": self.sample.metadata["tensor_sha256"],
            "fixed_evaluation_seed": self.config["fixed_evaluation_seed"],
            "fixed_evaluation_noise_sha256": tensor_sha256(self.fixed_batch.noise),
            "resume_contract": self._resume_contract(),
            "protocol": {
                "layout": self.config["layout_version"],
                "mask": self.config["mask_version"],
                "camera": self.config["camera_protocol"],
                "rope": self.config["rope_protocol"],
            },
            "git": git_identity(self.config["project_root"]),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(self.device),
            "wandb": {
                "enabled": self.wandb.enabled,
                "mode": self.wandb.mode,
                "started": False,
            },
        }
        self.run_manifest_path = self.output_dir / "run_manifest.json"
        with self.run_manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, default=_json_value)
            handle.write("\n")
        resolved_path = self.output_dir / "resolved_config.json"
        with resolved_path.open("w", encoding="utf-8") as handle:
            json.dump(
                resolved_config_for_json(self.config),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

    def _update_manifest_wandb(self, identity: Mapping[str, Any]) -> None:
        with self.run_manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["wandb"] = dict(identity)
        temporary = self.run_manifest_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, default=_json_value)
            handle.write("\n")
        os.replace(temporary, self.run_manifest_path)

    def start_tracking(self) -> dict[str, Any]:
        identity = self.wandb.start(self.restored_wandb_identity)
        self._update_manifest_wandb(identity)
        return identity

    def finish_tracking(self, *, exit_code: int = 0) -> None:
        self.wandb.finish(exit_code=exit_code)

    def _forward(self, batch: FlowTrainingBatch) -> tuple[torch.Tensor, Any]:
        tensors = self._sample_tensors()
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.compute_dtype == torch.bfloat16,
        ):
            output = self.model(
                noisy_states=batch.noisy_latents,
                clean_states=tensors["world_latents"],
                noisy_timesteps=batch.timesteps,
                prompt_embedding=tensors["prompt_embedding"],
                viewmats=tensors["viewmats"],
                Ks=tensors["Ks"],
            )
            losses = flow_matching_losses(output.flow, batch.targets, batch.weights)
        return output.flow, losses

    def log_record(self, kind: str, record: dict[str, Any]) -> None:
        append_jsonl(
            self.raw_log_path,
            {"kind": kind, "record": record, "wall_time": time.time()},
        )
        self.wandb.log(kind, record, step=self.global_step)
        print(json.dumps(record, default=_json_value, sort_keys=True), flush=True)

    def train_step(self) -> dict[str, Any]:
        context = (
            debug_scope(self.debug_tracer)
            if self.debug_mode and active_debug_tracer() is not self.debug_tracer
            else nullcontext()
        )
        with context:
            return self._train_step_impl()

    def _train_step_impl(self) -> dict[str, Any]:
        tracer = self.debug_tracer if self.debug_mode else None
        if tracer is not None:
            tracer.set_runtime_phase("forward")
            trace_event(
                "step",
                "training.step_start",
                details={"step": self.global_step + 1, "fixed_total_steps": 5},
            )
        self.model.train()
        tensors = self._sample_tensors()
        training_batch = sample_flow_training_batch(
            tensors["world_latents"], self.model.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        if tracer is not None:
            trace_event(
                "optimizer",
                "optimizer.zero_grad",
                details={"set_to_none": True},
            )
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        prediction, losses = self._forward(training_batch)
        loss_for_backward = trace_tensor(
            "loss",
            "loss.scalar_for_backward",
            losses.total,
            details={
                "total": float(losses.total.detach()),
                "init": float(losses.init.detach()),
                "transition": float(losses.transition.detach()),
            },
            backward=True,
        )
        if not torch.isfinite(loss_for_backward):
            raise FloatingPointError(f"Non-finite loss at step {self.global_step}: {losses.total}")
        if tracer is not None:
            trace_event(
                "backward",
                "backward.start",
                tensors={"loss": loss_for_backward},
                details={"expected_order": "head -> block_29 -> ... -> block_00 -> embeddings"},
                phase="backward",
            )
            tracer.set_runtime_phase("backward")
        try:
            with debug_timer() as backward_timing:
                loss_for_backward.backward()
        finally:
            if tracer is not None:
                tracer.set_runtime_phase("optimizer")
        with debug_timer() as clip_timing:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float(self.config["max_grad_norm"])
            )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm: {grad_norm}")
        captured_parameters: dict[str, dict[str, Any]] = {}
        if tracer is not None:
            with debug_timer() as snapshot_timing:
                captured_parameters = capture_parameter_values(self.model.named_parameters())
            gradient_rows = [
                {"name": name, **entry} for name, entry in captured_parameters.items()
            ]
            trace_event(
                "backward",
                "backward.parameter_gradients",
                details={
                    **backward_timing,
                    **clip_timing,
                    **snapshot_timing,
                    "global_gradient_norm_before_clip": float(grad_norm),
                    "max_gradient_norm": float(self.config["max_grad_norm"]),
                    "parameter_count": len(gradient_rows),
                    "parameters": gradient_rows,
                },
                phase="backward",
            )
        with debug_timer() as optimizer_timing:
            self.optimizer.step()
            self.lr_scheduler.step()
        if tracer is not None:
            with debug_timer() as update_timing:
                updates = parameter_updates(captured_parameters, self.model.named_parameters())
            trace_event(
                "optimizer",
                "optimizer.adamw_parameter_update",
                details={
                    **optimizer_timing,
                    **update_timing,
                    "parameter_count": len(updates),
                    "first_values_changed_count": sum(
                        int(row["first_values_changed"]) for row in updates
                    ),
                    "parameters": updates,
                },
                phase="optimizer",
            )
        self.global_step += 1
        torch.cuda.synchronize(self.device)

        timestep_float = training_batch.timesteps.float()
        record = {
            "step": self.global_step,
            "loss": float(losses.total.detach()),
            "loss_init": float(losses.init.detach()),
            "loss_transition": float(losses.transition.detach()),
            "per_state_loss": losses.per_state.detach().cpu().tolist(),
            "gradient_norm": float(grad_norm),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "timestep_min": float(timestep_float.min()),
            "timestep_max": float(timestep_float.max()),
            "timestep_mean": float(timestep_float.mean()),
            "prediction_mean": float(prediction.detach().float().mean()),
            "prediction_std": float(prediction.detach().float().std()),
            "target_mean": float(training_batch.targets.float().mean()),
            "target_std": float(training_batch.targets.float().std()),
            "step_seconds": time.perf_counter() - started,
            "allocated_gib": torch.cuda.memory_allocated(self.device) / 2**30,
            "reserved_gib": torch.cuda.memory_reserved(self.device) / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
        }
        append_jsonl(self.metrics_path, record)
        if tracer is not None:
            tracer.set_runtime_phase("forward")
            trace_event(
                "step",
                "training.step_complete",
                tensors={"per_state_loss": losses.per_state.detach()},
                details=record,
            )
        return record

    @torch.no_grad()
    def evaluate_fixed(self) -> dict[str, Any]:
        self.model.eval()
        prediction, losses = self._forward(self.fixed_batch)
        clean_prediction = flow_to_clean(
            self.fixed_batch.noisy_latents,
            prediction,
            self.fixed_batch.timesteps,
            self.model.scheduler,
        )
        target = self._sample_tensors()["world_latents"]
        metrics = latent_metrics(clean_prediction, target)
        metrics.update(
            {
                "step": self.global_step,
                "fixed_flow_loss": float(losses.total.detach()),
                "fixed_init_loss": float(losses.init.detach()),
                "fixed_transition_loss": float(losses.transition.detach()),
                "evaluation_kind": "teacher_forced_fixed_noise_single_step",
                "fixed_evaluation_seed": self.config["fixed_evaluation_seed"],
            }
        )
        append_jsonl(self.eval_path, metrics)
        return metrics

    def _rng_state(self) -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        }

    @staticmethod
    def _restore_rng_state(state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        torch.cuda.set_rng_state_all(state["torch_cuda"])

    def save_checkpoint(self, name: str, metrics: dict[str, Any]) -> Path:
        path = self.output_dir / f"{name}.pt"
        temporary = path.with_suffix(".pt.tmp")
        state: dict[str, Any] = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "generator": {
                key: value.detach().cpu()
                for key, value in self.model.generator.state_dict().items()
            },
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "metrics": metrics,
            "rng": self._rng_state(),
            "resolved_config": resolved_config_for_json(self.config),
            "base_checkpoint": self.base_checkpoint_audit,
            "cache_preprocessing_hash": self.sample.metadata["preprocessing_hash"],
            "cache_tensor_sha256": self.sample.metadata["tensor_sha256"],
            "layout_name": self.model.layout.name,
            "resume_contract": self._resume_contract(),
            "wandb": self.wandb.identity or self.restored_wandb_identity,
        }
        if bool(self.config.get("checkpoint_include_optimizer", True)):
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(state, temporary)
        os.replace(temporary, path)
        self.wandb.checkpoint_saved(
            path,
            name=name,
            step=self.global_step,
            best_metric=self.best_metric,
            best_step=self.best_step,
        )
        return path

    def load_checkpoint(self, path: str | Path, *, load_optimizer: bool = True) -> None:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if state.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise RuntimeError(f"Unsupported checkpoint version: {state.get('checkpoint_version')}")
        if state.get("layout_name") != self.model.layout.name:
            raise RuntimeError("Checkpoint layout is incompatible")
        if state.get("cache_preprocessing_hash") != self.sample.metadata["preprocessing_hash"]:
            raise RuntimeError("Checkpoint cache identity is incompatible")
        if state.get("cache_tensor_sha256") != self.sample.metadata["tensor_sha256"]:
            raise RuntimeError("Checkpoint cache tensor hashes are incompatible")
        if state.get("resume_contract") != self._resume_contract():
            raise RuntimeError("Checkpoint training/architecture contract is incompatible")
        incompatible = self.model.generator.load_state_dict(state["generator"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected resume mismatch: {incompatible}")
        if load_optimizer and "optimizer" in state and "lr_scheduler" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        elif load_optimizer:
            print(
                "Checkpoint is generator-only; optimizer and scheduler start fresh.",
                flush=True,
            )
        self.global_step = int(state["global_step"])
        self.best_metric = float(state["best_metric"])
        self.best_step = int(state["best_step"])
        restored_wandb = state.get("wandb")
        self.restored_wandb_identity = (
            dict(restored_wandb) if isinstance(restored_wandb, Mapping) else None
        )
        self._restore_rng_state(state["rng"])

    def train(self, max_steps: int) -> None:
        exit_code = 1
        try:
            self.start_tracking()
            self._train_loop(max_steps)
            exit_code = 0
        finally:
            self.finish_tracking(exit_code=exit_code)

    def _train_loop(self, max_steps: int) -> None:
        if self.global_step == 0:
            initial = self.evaluate_fixed()
            if bool(self.config.get("save_initial_checkpoint", True)):
                self.best_metric = float(initial["latent_mse"])
                self.best_step = 0
                self.save_checkpoint("initial", initial)
            else:
                self.best_metric = math.inf
                self.best_step = -1
            self.log_record("initial_evaluation", initial)
        while self.global_step < max_steps:
            record = self.train_step()
            if self.global_step % int(self.config["log_every"]) == 0:
                self.log_record("train_step", record)
            should_evaluate = (
                self.global_step % int(self.config["eval_every"]) == 0
                or self.global_step == max_steps
            )
            best_at_this_step = False
            if should_evaluate:
                metrics = self.evaluate_fixed()
                self.log_record("fixed_evaluation", metrics)
                if metrics["latent_mse"] < self.best_metric:
                    self.best_metric = float(metrics["latent_mse"])
                    self.best_step = self.global_step
                    best_at_this_step = True
            should_save = (
                self.global_step % int(self.config["save_every"]) == 0
                or self.global_step == max_steps
            )
            if should_save:
                if bool(self.config.get("retain_step_checkpoints", False)):
                    checkpoint = self.save_checkpoint(
                        f"step_{self.global_step:06d}",
                        metrics if should_evaluate else record,
                    )
                    replace_with_link(checkpoint, self.output_dir / "latest.pt")
                    if best_at_this_step:
                        replace_with_link(checkpoint, self.output_dir / "best.pt")
                else:
                    self.save_checkpoint("latest", record)
                    if best_at_this_step:
                        self.save_checkpoint("best", metrics)
            elif best_at_this_step:
                self.save_checkpoint("best", metrics)
