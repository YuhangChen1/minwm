from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

from full_duplex.config import load_config


def _normalize_generator_state(checkpoint: object) -> tuple[dict[str, torch.Tensor], str]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}")
    top_keys = list(checkpoint)
    if top_keys != ["generator"]:
        raise KeyError(f"Expected exactly top-level ['generator'], got {top_keys}")
    state = checkpoint["generator"]
    if not isinstance(state, dict) or not all(torch.is_tensor(value) for value in state.values()):
        raise TypeError("generator must be a pure tensor state dict")
    fixed: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        normalized = key.replace("model._fsdp_wrapped_module.", "model.", 1)
        if normalized in fixed:
            raise KeyError(f"Duplicate normalized checkpoint key: {normalized}")
        fixed[normalized] = value
    return fixed, "generator"


def audit(config: dict, run_forward: bool = True) -> dict:
    root = Path(config["project_root"])
    sys.path.insert(0, str(root / "Wan21"))
    sys.path.insert(0, str(root / "shared"))
    from wan_utils.wan_wrapper import WanDiffusionWrapper

    checkpoint_path = Path(config["base_checkpoint"])
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    state, selected_key = _normalize_generator_state(checkpoint)
    dtype_counts = Counter(str(value.dtype) for value in state.values())
    checkpoint_elements = sum(value.numel() for value in state.values())
    print(f"[checkpoint] path={checkpoint_path} top_keys={list(checkpoint)} selected={selected_key}")
    print(
        f"[checkpoint] tensors={len(state)} elements={checkpoint_elements} "
        f"dtypes={dict(dtype_counts)}"
    )

    wrapper = WanDiffusionWrapper(
        model_name="Wan2.1-T2V-1.3B",
        timestep_shift=config["timestep_shift"],
        is_causal=True,
        local_attn_size=20,
        use_camera=True,
    )
    model_state = wrapper.state_dict()
    model_elements = sum(value.numel() for value in model_state.values())
    # Enumerate first, then reject anything at all before the strict load.
    incompatible = wrapper.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    print(f"[checkpoint] missing_keys({len(missing)})={missing}")
    print(f"[checkpoint] unexpected_keys({len(unexpected)})={unexpected}")
    if missing or unexpected:
        raise RuntimeError("Base checkpoint is not an exact model match")
    wrapper.load_state_dict(state, strict=True)
    loaded_elements = sum(model_state[key].numel() for key in state)
    loaded_ratio = loaded_elements / model_elements
    if loaded_ratio != 1.0 or checkpoint_elements != model_elements:
        raise RuntimeError(
            f"Strict keys loaded but element coverage differs: {loaded_elements}/{model_elements}"
        )
    print(f"[checkpoint] strict load succeeded; loaded_ratio={loaded_ratio:.9f}")

    result = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "top_level_keys": list(checkpoint),
        "selected_state_key": selected_key,
        "tensor_keys": len(state),
        "parameter_elements": checkpoint_elements,
        "dtype_counts": dict(dtype_counts),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "loaded_parameter_ratio": loaded_ratio,
        "model_spec": {
            "dim": wrapper.model.dim,
            "ffn_dim": wrapper.model.ffn_dim,
            "num_heads": wrapper.model.num_heads,
            "num_layers": wrapper.model.num_layers,
            "patch_size": list(wrapper.model.patch_size),
            "in_dim": wrapper.model.in_dim,
            "out_dim": wrapper.model.out_dim,
            "text_len": wrapper.model.text_len,
            "text_dim": wrapper.model.text_dim,
        },
    }

    if run_forward:
        cache_path = Path(config["cache_path"]) / "tensors.pt"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Run preencode.py before real forward: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        dtype = torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
        device = torch.device("cuda:0")
        del checkpoint, model_state
        wrapper = wrapper.to(device=device, dtype=dtype).eval()
        clean = cache["world_state_latents"][1:2].unsqueeze(0).to(device=device, dtype=dtype)
        prompt = cache["prompt_embedding"].unsqueeze(0).to(device=device, dtype=dtype)
        viewmats = cache["viewmats"][1:2].unsqueeze(0).to(device=device, dtype=dtype)
        Ks = cache["Ks"][1:2].unsqueeze(0).to(device=device, dtype=dtype)
        generator = torch.Generator(device="cpu").manual_seed(config["fixed_noise_seed"])
        noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
        timestep_index = 500
        timestep = wrapper.scheduler.timesteps[timestep_index].to(device=device, dtype=dtype).reshape(1, 1)
        noisy = wrapper.scheduler.add_noise(
            clean.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, clean.shape[:2])
        torch.cuda.reset_peak_memory_stats(device)
        forward_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
            flow, x0 = wrapper(
                noisy_image_or_video=noisy,
                conditional_dict={"prompt_embeds": prompt},
                timestep=timestep,
                clean_x=clean,
                aug_t=torch.zeros_like(timestep),
                viewmats=viewmats,
                Ks=Ks,
            )
        torch.cuda.synchronize()
        if flow.shape != clean.shape or x0.shape != clean.shape:
            raise ValueError(f"Forward shape mismatch flow={flow.shape}, x0={x0.shape}, clean={clean.shape}")
        if not torch.isfinite(flow).all() or not torch.isfinite(x0).all():
            raise FloatingPointError("Original model real forward produced NaN/Inf")
        result["real_forward"] = {
            "clean_shape": list(clean.shape),
            "clean_dtype": str(clean.dtype),
            "prompt_shape": list(prompt.shape),
            "prompt_dtype": str(prompt.dtype),
            "viewmats_shape": list(viewmats.shape),
            "Ks_shape": list(Ks.shape),
            "flow_shape": list(flow.shape),
            "flow_dtype": str(flow.dtype),
            "flow_finite": True,
            "x0_finite": True,
            "flow_min": float(flow.float().min()),
            "flow_max": float(flow.float().max()),
            "elapsed_seconds": time.perf_counter() - forward_started,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }
        print(f"[checkpoint] real forward {json.dumps(result['real_forward'], sort_keys=True)}")
    result["audit_elapsed_seconds"] = time.perf_counter() - started
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "checkpoint_audit.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[checkpoint] report={report_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full_duplex/configs/overfit.yaml")
    parser.add_argument("--no-forward", action="store_true")
    args = parser.parse_args()
    audit(load_config(args.config), run_forward=not args.no_forward)


if __name__ == "__main__":
    main()
