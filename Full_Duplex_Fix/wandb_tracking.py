from __future__ import annotations

import argparse
import json
import numbers
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


WANDB_MODES = ("online", "offline", "disabled")


def add_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wandb",
        dest="wandb_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Weights & Biases for this training process.",
    )
    parser.add_argument("--wandb-mode", choices=WANDB_MODES, default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--wandb-tags",
        default=None,
        help="Comma-separated tags appended to the configured tags.",
    )
    parser.add_argument("--wandb-dir", default=None)


def wandb_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    result = {}
    for key in (
        "wandb_enabled",
        "wandb_mode",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
        "wandb_run_id",
        "wandb_group",
        "wandb_dir",
    ):
        value = getattr(args, key, None)
        if value is not None:
            result[key] = value
    tags = getattr(args, "wandb_tags", None)
    if tags is not None:
        result["wandb_tags"] = [item.strip() for item in tags.split(",") if item.strip()]
    return result


def _flatten_value(target: dict[str, Any], name: str, value: Any) -> None:
    if isinstance(value, Mapping):
        for child_name, child_value in value.items():
            _flatten_value(target, f"{name}/{child_name}", child_value)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child_value in enumerate(value):
            _flatten_value(target, f"{name}/state_{index:02d}", child_value)
        return
    if isinstance(value, numbers.Number):
        target[name] = value


def flatten_wandb_metrics(kind: str, record: Mapping[str, Any], step: int) -> dict[str, Any]:
    namespace = "train" if kind in {"train_step", "smoke_train_step"} else "eval"
    payload: dict[str, Any] = {
        "trainer/global_step": int(step),
        "trainer/event": kind,
    }
    for name, value in record.items():
        if name == "step":
            continue
        _flatten_value(payload, f"{namespace}/{name}", value)
    if kind == "initial_evaluation":
        payload["eval/is_initial"] = 1
    return payload


def _public_wandb_config(config: Mapping[str, Any]) -> dict[str, Any]:
    forbidden_keys = {
        "api_key",
        "password",
        "secret",
        "access_token",
        "auth_token",
        "wandb_api_key",
    }
    return {
        key: value
        for key, value in config.items()
        if key.lower() not in forbidden_keys and not key.lower().endswith("_api_key")
    }


class WandbTracker:
    def __init__(
        self,
        config: Mapping[str, Any],
        output_dir: str | Path,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = dict(config)
        self.config.update(dict(overrides or {}))
        self.output_dir = Path(output_dir).resolve()
        self.enabled = bool(self.config.get("wandb_enabled", False))
        self.mode = str(self.config.get("wandb_mode", "online"))
        if self.mode not in WANDB_MODES:
            raise ValueError(f"Unsupported wandb mode: {self.mode}")
        if self.enabled and not str(self.config.get("wandb_project") or "").strip():
            raise ValueError("wandb_project must be non-empty when wandb is enabled")
        if self.mode == "disabled":
            self.enabled = False
        self.run = None
        self.identity: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        return self.run is not None

    def start(self, restored_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if not self.enabled:
            self.identity = {"enabled": False, "mode": "disabled"}
            return dict(self.identity)
        if self.run is not None:
            return dict(self.identity or {})

        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "wandb is enabled but not installed; install the repository requirements"
            ) from error

        restored_identity = dict(restored_identity or {})
        run_id = (
            self.config.get("wandb_run_id")
            or restored_identity.get("run_id")
            or wandb.util.generate_id()
        )
        wandb_dir = Path(
            self.config.get("wandb_dir") or self.output_dir / "wandb"
        ).expanduser().resolve()
        wandb_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs: dict[str, Any] = {
            "project": self.config["wandb_project"],
            "name": self.config.get("wandb_run_name"),
            "id": str(run_id),
            "resume": "allow",
            "mode": self.mode,
            "dir": str(wandb_dir),
            "config": _public_wandb_config(self.config),
            "job_type": self.config.get("wandb_job_type", "single_sample_overfit"),
            "tags": list(self.config.get("wandb_tags") or ()),
            "group": self.config.get("wandb_group"),
            "notes": self.config.get("wandb_notes"),
        }
        entity = self.config.get("wandb_entity")
        if entity:
            init_kwargs["entity"] = entity
        try:
            self.run = wandb.init(**init_kwargs)
        except Exception as error:
            raise RuntimeError(
                "wandb initialization failed; run `wandb login`, set WANDB_API_KEY, "
                "use --wandb-mode offline, or pass --no-wandb"
            ) from error
        if self.run is None:
            raise RuntimeError("wandb.init returned no run")

        self.run.define_metric("trainer/global_step")
        for namespace in ("train/*", "eval/*", "checkpoint/*"):
            self.run.define_metric(namespace, step_metric="trainer/global_step")
        run_directory = Path(self.run.dir).resolve().parent
        self.identity = {
            "enabled": True,
            "mode": self.mode,
            "run_id": self.run.id,
            "run_name": self.run.name,
            "project": self.run.project,
            "entity": self.run.entity,
            "url": self.run.url,
            "wandb_root": str(wandb_dir),
            "run_directory": str(run_directory),
            "resumed": bool(getattr(self.run, "resumed", False)),
        }
        self._write_identity()
        return dict(self.identity)

    def _write_identity(self) -> None:
        if self.identity is None:
            return
        path = self.output_dir / "wandb_run.json"
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.identity, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)

    def log(self, kind: str, record: Mapping[str, Any], *, step: int) -> None:
        if self.run is None:
            return
        self.run.log(flatten_wandb_metrics(kind, record, step))

    def checkpoint_saved(
        self,
        path: str | Path,
        *,
        name: str,
        step: int,
        best_metric: float,
        best_step: int,
    ) -> None:
        if self.run is None:
            return
        path = Path(path).resolve()
        self.run.summary["checkpoint/latest_name"] = name
        self.run.summary["checkpoint/latest_path"] = str(path)
        self.run.summary["checkpoint/latest_step"] = int(step)
        self.run.summary["checkpoint/best_metric"] = float(best_metric)
        self.run.summary["checkpoint/best_step"] = int(best_step)
        self.run.log(
            {
                "trainer/global_step": int(step),
                f"checkpoint/{name}_saved": 1,
                "checkpoint/file_size_gib": path.stat().st_size / 2**30,
            }
        )
        if bool(self.config.get("wandb_log_checkpoints", False)):
            import wandb

            artifact = wandb.Artifact(
                name=f"{self.run.id}-{name}",
                type="model",
                metadata={"global_step": int(step), "best_metric": float(best_metric)},
            )
            artifact.add_file(str(path))
            self.run.log_artifact(artifact, aliases=[name, f"step-{step}"])

    def finish(self, *, exit_code: int = 0) -> None:
        if self.run is None:
            return
        self.run.finish(exit_code=exit_code)
        self.run = None
