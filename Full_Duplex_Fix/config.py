from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str, project_root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a YAML mapping in {config_path}")

    project_root = Path(raw.get("project_root", PROJECT_ROOT)).expanduser().resolve()
    config = dict(raw)
    config["project_root"] = str(project_root)
    config["config_path"] = str(config_path)

    path_keys = (
        "base_checkpoint",
        "vae_checkpoint",
        "t5_checkpoint",
        "t5_tokenizer",
        "video_path",
        "metadata_path",
        "action_manifest",
        "input_manifest",
        "cache_path",
        "dataset_cache_path",
        "output_dir",
        "wandb_dir",
    )
    for key in path_keys:
        if key in config:
            config[key] = _resolve_path(str(config[key]), project_root)

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    training_mode = str(config.get("training_mode", "single_sample"))
    if training_mode not in {"single_sample", "multi_sample"}:
        raise ValueError("training_mode must be single_sample or multi_sample")

    exact = {
        "num_states": 20,
        "num_noisy_spans": 20,
        "num_clean_spans": 20,
        "num_total_spans": 40,
        "tokens_per_span": 1560,
        "sequence_length": 62400,
        "latent_channels": 16,
        "latent_height": 60,
        "latent_width": 104,
        "patch_height": 30,
        "patch_width": 52,
        "num_transformer_blocks": 30,
        "num_frame_per_block": 1,
        "spatial_token_stride": 1,
        "source_frames": 77,
        "source_fps": 24,
        "frames_per_transition": 4,
        "target_height": 480,
        "target_width": 832,
        "model_dim": 1536,
        "num_heads": 12,
        "head_dim": 128,
        "text_length": 512,
        "text_dim": 4096,
        "batch_size": 1,
        "local_attention_states": 20,
        "num_train_timesteps": 1000,
    }
    for key, expected in exact.items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(f"{key} must be {expected}, got {actual}")

    if tuple(config.get("patch_size", ())) != (1, 2, 2):
        raise ValueError("patch_size must remain [1, 2, 2]")
    if config.get("generator_checkpoint_stage") != "ar_diffusion_tf":
        raise ValueError("The primary experiment must initialize from ar_diffusion_tf")
    if config.get("learning_rate", 0) <= 0:
        raise ValueError("learning_rate must be positive")
    if config.get("sampling_steps", 0) <= 0:
        raise ValueError("sampling_steps must be positive")
    if config.get("mixed_precision") not in {"bf16", "fp32"}:
        raise ValueError("mixed_precision must be bf16 or fp32")
    if config.get("fsdp_enabled") is not False:
        raise ValueError("fsdp_enabled must be false for the verified H200 paths")
    if config.get("sequence_parallel_size") != 1:
        raise ValueError("sequence_parallel_size must be 1")
    if training_mode == "single_sample":
        if config.get("distributed_backend") != "single_gpu" or config.get("world_size") != 1:
            raise ValueError("single_sample requires single_gpu and world_size=1")
    elif config.get("distributed_backend") != "ddp":
        raise ValueError("multi_sample requires distributed_backend=ddp")
    elif not isinstance(config.get("world_size"), int) or config["world_size"] <= 0:
        raise ValueError("multi_sample world_size must be a positive integer")
    if config.get("lr_scheduler") != "constant":
        raise ValueError("lr_scheduler must be constant")

    accumulation = config.get("gradient_accumulation_steps")
    if not isinstance(accumulation, int) or accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be a positive integer")

    for key in ("max_steps", "save_every", "eval_every", "log_every"):
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")

    if not isinstance(config.get("wandb_enabled"), bool):
        raise ValueError("wandb_enabled must be true or false")
    if config.get("wandb_mode") not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline, or disabled")
    if config["wandb_enabled"] and not str(config.get("wandb_project") or "").strip():
        raise ValueError("wandb_project must be non-empty when wandb is enabled")
    for key in (
        "wandb_entity",
        "wandb_run_name",
        "wandb_run_id",
        "wandb_group",
        "wandb_job_type",
        "wandb_notes",
    ):
        if config.get(key) is not None and not isinstance(config[key], str):
            raise ValueError(f"{key} must be null or a string")
    tags = config.get("wandb_tags")
    if not isinstance(tags, list) or not all(
        isinstance(tag, str) and tag.strip() for tag in tags
    ):
        raise ValueError("wandb_tags must be a list of non-empty strings")
    if not isinstance(config.get("wandb_log_checkpoints"), bool):
        raise ValueError("wandb_log_checkpoints must be true or false")
    for key in (
        "checkpoint_include_optimizer",
        "save_initial_checkpoint",
        "retain_step_checkpoints",
    ):
        if not isinstance(config.get(key, False), bool):
            raise ValueError(f"{key} must be true or false")
    secret_keys = {
        "api_key",
        "password",
        "secret",
        "access_token",
        "auth_token",
        "wandb_api_key",
    }
    configured_secrets = [
        key
        for key in config
        if key.lower() in secret_keys or key.lower().endswith("_api_key")
    ]
    if configured_secrets:
        raise ValueError(
            "Credentials must not be stored in the config; use WANDB_API_KEY or "
            f"`wandb login` instead (found: {configured_secrets})"
        )

    required_files = [
        "base_checkpoint",
        "vae_checkpoint",
        "t5_checkpoint",
        "t5_tokenizer",
    ]
    if training_mode == "single_sample":
        required_files.extend(("video_path", "metadata_path", "action_manifest"))
    else:
        required_files.append("input_manifest")
        if not str(config.get("dataset_cache_path") or "").strip():
            raise ValueError("multi_sample requires dataset_cache_path")
        expected_size = config.get("expected_dataset_size")
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise ValueError("expected_dataset_size must be a positive integer")
        validation_size = config.get("validation_size")
        if not isinstance(validation_size, int) or not 0 < validation_size < expected_size:
            raise ValueError("validation_size must be between 1 and expected_dataset_size - 1")
        if not isinstance(config.get("train_all_samples"), bool):
            raise ValueError("train_all_samples must be true or false")
        eval_samples = config.get("eval_num_samples")
        if not isinstance(eval_samples, int) or not 0 < eval_samples <= validation_size:
            raise ValueError("eval_num_samples must be between 1 and validation_size")
        workers = config.get("dataloader_num_workers")
        if not isinstance(workers, int) or workers < 0:
            raise ValueError("dataloader_num_workers must be a non-negative integer")
        effective_batch = config["batch_size"] * config["world_size"] * accumulation
        if config.get("total_batch_size") != effective_batch:
            raise ValueError(
                f"total_batch_size must equal batch_size*world_size*accumulation={effective_batch}"
            )
    missing = [key for key in required_files if not Path(config[key]).exists()]
    if missing:
        details = ", ".join(f"{key}={config[key]}" for key in missing)
        raise FileNotFoundError(f"Missing configured inputs: {details}")


def resolved_config_for_json(config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result
