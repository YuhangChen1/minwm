from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .checkpoint import load_strict_generator
from .checkpoint_io import replace_with_link
from .config import resolved_config_for_json
from .dataset_cache import (
    CACHE_MANIFEST_NAME,
    PreencodedVideoDataset,
    deterministic_split,
    sha256_file,
)
from .flow import flow_matching_losses, flow_to_clean, sample_flow_training_batch
from .model import InterleavedWanAdapter
from .training import CHECKPOINT_VERSION, append_jsonl, git_identity, set_seed
from .wandb_tracking import WandbTracker


class MultiSampleTrainer:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        wandb_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("MultiSampleTrainer must run under initialized torch.distributed")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        if self.world_size != int(config["world_size"]):
            raise RuntimeError(
                f"Configured world_size={config['world_size']}, actual={self.world_size}"
            )
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.config = dict(config)
        overrides = dict(wandb_overrides or {})
        extra_tags = overrides.pop("wandb_tags", None)
        self.config.update(overrides)
        if extra_tags is not None:
            self.config["wandb_tags"] = list(
                dict.fromkeys([*self.config.get("wandb_tags", ()), *extra_tags])
            )
        config = self.config
        set_seed(int(config["seed"]) + self.rank)
        self.compute_dtype = (
            torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
        )

        split_train_indices, validation_indices = deterministic_split(
            int(config["expected_dataset_size"]),
            validation_size=int(config["validation_size"]),
            seed=int(config["dataset_split_seed"]),
        )
        self.validation_is_heldout = not bool(config["train_all_samples"])
        self.train_indices = (
            split_train_indices
            if self.validation_is_heldout
            else list(range(int(config["expected_dataset_size"])))
        )
        self.validation_indices = validation_indices
        self.train_dataset = PreencodedVideoDataset(
            config["dataset_cache_path"],
            indices=self.train_indices,
            expected_count=int(config["expected_dataset_size"]),
        )
        self.validation_dataset = (
            PreencodedVideoDataset(
                config["dataset_cache_path"],
                indices=validation_indices[: int(config["eval_num_samples"])],
                expected_count=int(config["expected_dataset_size"]),
            )
            if self.rank == 0
            else None
        )
        self.sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=int(config["dataset_split_seed"]),
            drop_last=True,
        )
        loader_kwargs: dict[str, Any] = {
            "dataset": self.train_dataset,
            "batch_size": int(config["batch_size"]),
            "sampler": self.sampler,
            "num_workers": int(config["dataloader_num_workers"]),
            "pin_memory": True,
            "drop_last": True,
        }
        if int(config["dataloader_num_workers"]) > 0:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
        self.loader = DataLoader(**loader_kwargs)
        self.loader_epoch = 0
        self.sampler.set_epoch(self.loader_epoch)
        self.loader_iterator = iter(self.loader)

        generator, self.base_checkpoint_audit = load_strict_generator(
            config, device=self.device, dtype=torch.float32
        )
        self.module = InterleavedWanAdapter(
            generator,
            gradient_checkpointing=bool(config["gradient_checkpointing"]),
        ).to(self.device)
        self.model = DistributedDataParallel(
            self.module,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in self.module.parameters() if parameter.requires_grad],
            lr=float(config["learning_rate"]),
            betas=(float(config["beta1"]), float(config["beta2"])),
            weight_decay=float(config["weight_decay"]),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda _: 1.0
        )
        self.global_step = 0
        self.best_metric = math.inf
        self.best_step = -1
        self.restored_wandb_identity: dict[str, Any] | None = None

        self.output_dir = Path(config["output_dir"])
        if self.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.eval_path = self.output_dir / "evaluation.jsonl"
        self.raw_log_path = self.output_dir / "raw_log.jsonl"
        self.wandb = WandbTracker(config, self.output_dir) if self.rank == 0 else None
        if self.rank == 0:
            self._write_run_manifest()

    def _write_run_manifest(self) -> None:
        cache_manifest = Path(self.config["dataset_cache_path"]) / CACHE_MANIFEST_NAME
        manifest = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "training_mode": "multi_sample",
            "resolved_config": resolved_config_for_json(self.config),
            "base_checkpoint": self.base_checkpoint_audit,
            "parameters": self.module.parameter_manifest(),
            "layout": {
                "name": self.module.layout.name,
                "labels": self.module.layout.labels(),
                "spans": [asdict(span) for span in self.module.layout.spans],
                "sequence_length": self.module.layout.sequence_length,
            },
            "dataset": {
                "cache_manifest": str(cache_manifest.resolve()),
                "cache_manifest_sha256": sha256_file(cache_manifest),
                "train_count": len(self.train_indices),
                "validation_count": len(self.validation_indices),
                "validation_is_heldout": self.validation_is_heldout,
                "validation_role": (
                    "heldout_validation"
                    if self.validation_is_heldout
                    else "fixed_training_monitor_subset"
                ),
                "train_indices": self.train_indices,
                "validation_indices": self.validation_indices,
            },
            "global_batch_size": int(self.config["batch_size"])
            * self.world_size
            * int(self.config["gradient_accumulation_steps"]),
            "git": git_identity(self.config["project_root"]),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "devices": self.world_size,
        }
        path = self.output_dir / "run_manifest.json"
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        with (self.output_dir / "resolved_config.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                resolved_config_for_json(self.config),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

    def _next_batch(self) -> dict[str, Any]:
        try:
            return next(self.loader_iterator)
        except StopIteration:
            self.loader_epoch += 1
            self.sampler.set_epoch(self.loader_epoch)
            self.loader_iterator = iter(self.loader)
            return next(self.loader_iterator)

    def _to_device(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            "world_latents": batch["world_latents"].to(
                self.device, dtype=self.compute_dtype, non_blocking=True
            ),
            "prompt_embedding": batch["prompt_embedding"].to(
                self.device, dtype=self.compute_dtype, non_blocking=True
            ),
            "viewmats": batch["viewmats"].to(
                self.device, dtype=self.compute_dtype, non_blocking=True
            ),
            "Ks": batch["Ks"].to(
                self.device, dtype=self.compute_dtype, non_blocking=True
            ),
            "sample_index": batch["sample_index"].to(self.device, non_blocking=True),
        }

    def _forward(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        distributed: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[Any, torch.Tensor]:
        flow_batch = sample_flow_training_batch(
            tensors["world_latents"], self.module.scheduler, generator=generator
        )
        active_model = self.model if distributed else self.module
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.compute_dtype == torch.bfloat16,
        ):
            output = active_model(
                noisy_states=flow_batch.noisy_latents,
                clean_states=tensors["world_latents"],
                noisy_timesteps=flow_batch.timesteps,
                prompt_embedding=tensors["prompt_embedding"],
                viewmats=tensors["viewmats"],
                Ks=tensors["Ks"],
            )
            losses = flow_matching_losses(
                output.flow, flow_batch.targets, flow_batch.weights
            )
        return (flow_batch, output.flow, losses), tensors["sample_index"]

    def _reduce_mean(self, value: torch.Tensor) -> torch.Tensor:
        value = value.detach().clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value / self.world_size

    def train_step(self) -> dict[str, Any]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        accumulation = int(self.config["gradient_accumulation_steps"])
        metric_sum = torch.zeros(23, device=self.device, dtype=torch.float32)
        sample_indices: list[int] = []

        for micro_step in range(accumulation):
            tensors = self._to_device(self._next_batch())
            synchronize = micro_step == accumulation - 1
            context = nullcontext() if synchronize else self.model.no_sync()
            with context:
                (flow_batch, prediction, losses), batch_indices = self._forward(
                    tensors, distributed=True
                )
                scaled_loss = losses.total / accumulation
                if not torch.isfinite(scaled_loss):
                    raise FloatingPointError(
                        f"Non-finite loss at optimizer step {self.global_step + 1}"
                    )
                scaled_loss.backward()
            metric_sum[:3] += torch.stack(
                (losses.total.detach(), losses.init.detach(), losses.transition.detach())
            ).float()
            metric_sum[3:] += losses.per_state.detach().float()
            sample_indices.extend(int(value) for value in batch_indices.cpu().tolist())

        metric_sum /= accumulation
        metric_mean = self._reduce_mean(metric_sum)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.module.parameters(), float(self.config["max_grad_norm"])
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm: {grad_norm}")
        self.optimizer.step()
        self.lr_scheduler.step()
        self.global_step += 1
        torch.cuda.synchronize(self.device)
        elapsed = torch.tensor(time.perf_counter() - started, device=self.device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        record = {
            "step": self.global_step,
            "loss": float(metric_mean[0]),
            "loss_init": float(metric_mean[1]),
            "loss_transition": float(metric_mean[2]),
            "per_state_loss": metric_mean[3:].cpu().tolist(),
            "gradient_norm": float(grad_norm),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "optimizer_step_seconds": float(elapsed),
            "global_batch_size": int(self.config["batch_size"])
            * self.world_size
            * accumulation,
            "rank0_sample_indices": sample_indices if self.rank == 0 else [],
            "rank0_peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
        }
        if self.rank == 0:
            append_jsonl(self.metrics_path, record)
        return record

    @torch.no_grad()
    def evaluate_fixed(self) -> dict[str, Any]:
        if self.rank != 0 or self.validation_dataset is None:
            raise RuntimeError("Fixed evaluation is rank-0 only")
        self.module.eval()
        loss_total = torch.zeros(23, device=self.device, dtype=torch.float64)
        latent_mse = torch.zeros(20, device=self.device, dtype=torch.float64)
        latent_cosine = torch.zeros(20, device=self.device, dtype=torch.float64)
        evaluated_indices: list[int] = []
        for item in self.validation_dataset:
            batched = {
                name: (
                    value.unsqueeze(0)
                    if isinstance(value, torch.Tensor)
                    else torch.tensor([value], dtype=torch.long)
                )
                for name, value in item.items()
            }
            tensors = self._to_device(batched)
            sample_index = int(tensors["sample_index"].item())
            generator = torch.Generator(device=self.device).manual_seed(
                int(self.config["fixed_evaluation_seed"]) + sample_index
            )
            (flow_batch, prediction, losses), _ = self._forward(
                tensors, distributed=False, generator=generator
            )
            loss_total[:3] += torch.stack(
                (losses.total, losses.init, losses.transition)
            ).double()
            loss_total[3:] += losses.per_state.double()
            clean_prediction = flow_to_clean(
                flow_batch.noisy_latents,
                prediction,
                flow_batch.timesteps,
                self.module.scheduler,
            ).float()
            target = tensors["world_latents"].float()
            delta = clean_prediction - target
            latent_mse += delta.square().mean(dim=(0, 2, 3, 4)).double()
            latent_cosine += torch.nn.functional.cosine_similarity(
                clean_prediction.flatten(2), target.flatten(2), dim=2
            ).mean(dim=0).double()
            evaluated_indices.append(sample_index)
        count = len(self.validation_dataset)
        loss_total /= count
        latent_mse /= count
        latent_cosine /= count
        return {
            "step": self.global_step,
            "fixed_flow_loss": float(loss_total[0]),
            "fixed_init_loss": float(loss_total[1]),
            "fixed_transition_loss": float(loss_total[2]),
            "fixed_per_state_flow_loss": loss_total[3:].cpu().tolist(),
            "latent_mse": float(latent_mse.mean()),
            "latent_cosine": float(latent_cosine.mean()),
            "per_state_mse": latent_mse.cpu().tolist(),
            "per_state_cosine": latent_cosine.cpu().tolist(),
            "evaluation_kind": (
                "heldout_teacher_forced_fixed_noise_single_step"
                if self.validation_is_heldout
                else "training_subset_teacher_forced_fixed_noise_single_step"
            ),
            "evaluated_sample_indices": evaluated_indices,
            "fixed_evaluation_seed": int(self.config["fixed_evaluation_seed"]),
        }

    def log_record(self, kind: str, record: dict[str, Any]) -> None:
        if self.rank != 0:
            return
        append_jsonl(
            self.raw_log_path,
            {"kind": kind, "record": record, "wall_time": time.time()},
        )
        if self.wandb is not None:
            self.wandb.log(kind, record, step=self.global_step)
        print(json.dumps(record, sort_keys=True), flush=True)

    def save_checkpoint(self, metrics: dict[str, Any]) -> Path:
        if self.rank != 0:
            raise RuntimeError("Only rank 0 may save checkpoints")
        path = self.output_dir / f"step_{self.global_step:06d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        cache_manifest = Path(self.config["dataset_cache_path"]) / CACHE_MANIFEST_NAME
        state: dict[str, Any] = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "training_mode": "multi_sample",
            "generator": {
                key: value.detach().cpu()
                for key, value in self.module.generator.state_dict().items()
            },
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "metrics": metrics,
            "resolved_config": resolved_config_for_json(self.config),
            "base_checkpoint": self.base_checkpoint_audit,
            "layout_name": self.module.layout.name,
            "cache_manifest_sha256": sha256_file(cache_manifest),
            "wandb": self.wandb.identity if self.wandb is not None else None,
        }
        if bool(self.config.get("checkpoint_include_optimizer", False)):
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(state, temporary)
        os.replace(temporary, path)
        replace_with_link(path, self.output_dir / "latest.pt")
        if self.best_step == self.global_step:
            replace_with_link(path, self.output_dir / "best.pt")
        if self.wandb is not None:
            self.wandb.checkpoint_saved(
                path,
                name=path.stem,
                step=self.global_step,
                best_metric=self.best_metric,
                best_step=self.best_step,
            )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if state.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise RuntimeError(f"Unsupported checkpoint version: {state.get('checkpoint_version')}")
        if state.get("layout_name") != self.module.layout.name:
            raise RuntimeError("Checkpoint layout is incompatible")
        cache_manifest = Path(self.config["dataset_cache_path"]) / CACHE_MANIFEST_NAME
        if state.get("cache_manifest_sha256") != sha256_file(cache_manifest):
            raise RuntimeError("Checkpoint dataset cache manifest is incompatible")
        incompatible = self.module.generator.load_state_dict(state["generator"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected resume mismatch: {incompatible}")
        if "optimizer" in state and "lr_scheduler" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        elif self.rank == 0:
            print(
                "Checkpoint is generator-only; optimizer and scheduler start fresh.",
                flush=True,
            )
        self.global_step = int(state["global_step"])
        self.best_metric = float(state.get("best_metric", math.inf))
        self.best_step = int(state.get("best_step", -1))
        restored = state.get("wandb")
        self.restored_wandb_identity = dict(restored) if isinstance(restored, Mapping) else None
        dist.barrier()

    def train(self, max_steps: int) -> None:
        exit_code = 1
        try:
            if self.rank == 0 and self.wandb is not None:
                self.wandb.start(self.restored_wandb_identity)
            if self.global_step == 0:
                dist.barrier()
                if self.rank == 0:
                    initial = self.evaluate_fixed()
                    self.best_metric = math.inf
                    self.best_step = -1
                    append_jsonl(self.eval_path, initial)
                    self.log_record("initial_evaluation", initial)
                dist.barrier()

            while self.global_step < max_steps:
                record = self.train_step()
                if self.global_step % int(self.config["log_every"]) == 0:
                    self.log_record("train_step", record)
                should_evaluate = (
                    self.global_step % int(self.config["eval_every"]) == 0
                    or self.global_step == max_steps
                )
                evaluation = None
                if should_evaluate:
                    dist.barrier()
                    if self.rank == 0:
                        evaluation = self.evaluate_fixed()
                        append_jsonl(self.eval_path, evaluation)
                        self.log_record("fixed_evaluation", evaluation)
                    dist.barrier()
                should_save = (
                    self.global_step % int(self.config["save_every"]) == 0
                    or self.global_step == max_steps
                )
                if should_save:
                    dist.barrier()
                    if self.rank == 0:
                        if (
                            evaluation is not None
                            and evaluation["latent_mse"] < self.best_metric
                        ):
                            self.best_metric = float(evaluation["latent_mse"])
                            self.best_step = self.global_step
                        self.save_checkpoint(evaluation or record)
                    dist.barrier()
            exit_code = 0
        finally:
            if self.rank == 0 and self.wandb is not None:
                self.wandb.finish(exit_code=exit_code)
