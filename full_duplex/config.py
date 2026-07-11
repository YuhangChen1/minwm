from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL = {
    "base_checkpoint",
    "vae_checkpoint",
    "t5_checkpoint",
    "t5_tokenizer",
    "dataset_path",
    "video_path",
    "action_manifest",
    "cache_path",
    "output_dir",
    "num_micro_turns",
    "num_denoising_steps",
    "num_frame_per_turn",
    "max_time_index",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "gradient_accumulation_steps",
    "max_grad_norm",
    "mixed_precision",
    "gradient_checkpointing",
    "seed",
    "lambda_flow",
    "lambda_state",
    "lambda_camera",
    "lambda_translation",
    "lambda_rotation",
    "lambda_intrinsics",
    "save_every",
    "log_every",
    "max_steps",
    "teacher_forcing_ratio",
    "detach_between_turns",
}


def _canonical(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def config_hash(config: dict[str, Any], keys: list[str] | None = None) -> str:
    selected = config if keys is None else {key: config[key] for key in keys}
    payload = json.dumps(_canonical(selected), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def refresh_training_config_hash(config: dict[str, Any]) -> str:
    """Refresh the hash after command-line overrides without self-hashing."""
    payload = {key: value for key, value in config.items() if key != "training_config_hash"}
    digest = config_hash(payload)
    config["training_config_hash"] = digest
    return digest


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(config).__name__}")
    missing = sorted(REQUIRED_TOP_LEVEL - set(config))
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    root = Path(config.get("project_root", path.parents[2])).resolve()
    config["project_root"] = str(root)
    path_keys = (
        "base_checkpoint",
        "vae_checkpoint",
        "t5_checkpoint",
        "t5_tokenizer",
        "dataset_path",
        "video_path",
        "action_manifest",
        "cache_path",
        "output_dir",
    )
    for key in path_keys:
        value = Path(config[key])
        config[key] = str((root / value).resolve() if not value.is_absolute() else value.resolve())

    if config["mixed_precision"] not in ("bf16", "fp32"):
        raise ValueError("mixed_precision must be 'bf16' or 'fp32'")
    if config["teacher_forcing_ratio"] != 0.0:
        raise ValueError("Default Full-Duplex rollout requires teacher_forcing_ratio: 0.0")
    if config["detach_between_turns"]:
        raise ValueError("Cross-turn BPTT requires detach_between_turns: false")
    if config["num_micro_turns"] > config["max_time_index"] + 1:
        raise ValueError("num_micro_turns exceeds the TIME_INDEX vocabulary")
    config["config_path"] = str(path)
    refresh_training_config_hash(config)
    return config
