from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from full_duplex.config import load_config


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full_duplex/configs/overfit.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("full_duplex/outputs/smallest_000000/environment_audit.json"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["project_root"])
    for path in (root, root / "Wan21", root / "shared"):
        sys.path.insert(0, str(path))

    import_results: dict[str, str] = {}
    for module in (
        "wan_utils.wan_wrapper",
        "wan_utils.dataset",
        "wan.modules.model",
        "wan.modules.causal_model",
        "wan.modules.prope",
        "wan.modules.vae",
    ):
        imported = importlib.import_module(module)
        import_results[module] = str(Path(imported.__file__).resolve())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    properties = torch.cuda.get_device_properties(0)
    paths = {
        key: {
            "path": config[key],
            "exists": Path(config[key]).exists(),
            "is_file": Path(config[key]).is_file(),
        }
        for key in (
            "base_checkpoint",
            "vae_checkpoint",
            "t5_checkpoint",
            "dataset_path",
            "video_path",
            "action_manifest",
        )
    }
    if not all(entry["exists"] for entry in paths.values()):
        raise FileNotFoundError(f"Required paths missing: {paths}")

    report: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "requested_conda_environment": "/hyperai/home/conda_envs/minwm",
        "environment_matches_request": Path(sys.prefix).resolve()
        == Path("/hyperai/home/conda_envs/minwm").resolve(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "gpu": {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        },
        "dependencies": {
            name: _version(name)
            for name in (
                "decord",
                "diffusers",
                "transformers",
                "safetensors",
                "PyYAML",
                "numpy",
                "scipy",
            )
        },
        "paths": paths,
        "original_project_import_test": import_results,
        "cwd": os.getcwd(),
    }
    if "H100" not in properties.name:
        raise RuntimeError(f"Expected H100, found {properties.name}")
    if not report["bf16_supported"]:
        raise RuntimeError("GPU does not report bf16 support")
    if not report["environment_matches_request"]:
        raise RuntimeError(
            f"Interpreter prefix {sys.prefix} is not the requested conda environment"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
