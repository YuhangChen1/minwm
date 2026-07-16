from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: str | Path, chunk_size: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path: str | Path, *, compute_hash: bool) -> dict[str, Any]:
    path = Path(path).resolve()
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if compute_hash:
        result["sha256"] = sha256_file(path)
    return result


def verify_checkpoint(path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    identity = checkpoint_identity(path, compute_hash=bool(config["verify_checkpoint_hash"]))
    expected_size = int(config["expected_base_checkpoint_size"])
    if identity["size"] != expected_size:
        raise RuntimeError(
            f"AR checkpoint size mismatch: expected {expected_size}, got {identity['size']}"
        )
    if config["verify_checkpoint_hash"]:
        expected_hash = str(config["expected_base_checkpoint_sha256"])
        if identity["sha256"] != expected_hash:
            raise RuntimeError(
                f"AR checkpoint SHA256 mismatch: expected {expected_hash}, got {identity['sha256']}"
            )
    return identity


def normalize_generator_state_dict(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "generator" in state:
        state = state["generator"]
    elif "model" in state:
        state = state["model"]
    elif "generator_ema" in state:
        state = state["generator_ema"]
    if not isinstance(state, dict):
        raise TypeError("Generator checkpoint payload must be a state dict")

    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for prefix in ("_checkpoint_wrapped_module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key.removeprefix(prefix)
        if key.startswith("model._fsdp_wrapped_module."):
            key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
        normalized[key] = value
    return normalized


def load_strict_generator(
    config: dict[str, Any],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    project_root = Path(config["project_root"])
    wan_root = project_root / "Wan21"
    shared_root = project_root / "shared"
    for path in (str(wan_root), str(shared_root), str(project_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from wan_utils.wan_wrapper import WanDiffusionWrapper

    identity = verify_checkpoint(config["base_checkpoint"], config)
    generator = WanDiffusionWrapper(
        model_name="Wan2.1-T2V-1.3B",
        timestep_shift=float(config["timestep_shift"]),
        is_causal=True,
        local_attn_size=int(config["local_attention_states"]),
        sink_size=0,
        use_camera=True,
    )
    raw = torch.load(
        config["base_checkpoint"], map_location="cpu", mmap=True, weights_only=True
    )
    state = normalize_generator_state_dict(raw)
    incompatible = generator.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected strict-load mismatch: {incompatible}")
    if len(state) != len(generator.state_dict()):
        raise RuntimeError(
            f"Strict load count mismatch: checkpoint={len(state)}, model={len(generator.state_dict())}"
        )
    equality_keys = (
        "model.patch_embedding.weight",
        "model.blocks.0.self_attn.q.weight",
        "model.blocks.0.self_attn.prope_o.weight",
        "model.blocks.29.self_attn.prope_o.weight",
        "model.head.head.weight",
    )
    loaded_state = generator.state_dict()
    unequal = [key for key in equality_keys if not torch.equal(loaded_state[key], state[key])]
    if unequal:
        raise RuntimeError(f"Loaded tensors differ from the checkpoint: {unequal}")
    del raw, state
    generator = generator.to(device=device, dtype=dtype)
    generator.train()
    identity.update(
        {
            "strict_load": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "parameter_count": sum(parameter.numel() for parameter in generator.parameters()),
            "tensor_count": len(generator.state_dict()),
            "sampled_tensor_equality": list(equality_keys),
            "prope_layers": sum(
                1 for name, _ in generator.named_modules() if name.endswith("self_attn.prope_o")
            ),
        }
    )
    return generator, identity
