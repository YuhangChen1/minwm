from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config
from .training import OverfitTrainer


PARITY_KEYS = (
    "latent_mse",
    "latent_cosine",
    "bootstrap_mse",
    "transition_mse",
    "fixed_flow_loss",
    "fixed_init_loss",
    "fixed_transition_loss",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Full_Duplex_Fix/configs/overfit.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        default="Full_Duplex_Fix/outputs/smallest_000000/checkpoint_parity.json",
    )
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    saved = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    trainer = OverfitTrainer(load_config(args.config), device=torch.device(args.device))
    trainer.load_checkpoint(checkpoint_path, load_optimizer=True)
    fresh = trainer.evaluate_fixed()
    reference = saved.get("metrics", {})
    comparisons = {}
    for key in PARITY_KEYS:
        if key in reference:
            absolute_error = abs(float(fresh[key]) - float(reference[key]))
            comparisons[key] = {
                "saved": float(reference[key]),
                "fresh": float(fresh[key]),
                "absolute_error": absolute_error,
            }
            if absolute_error > args.tolerance:
                raise AssertionError(
                    f"Fresh reload parity failed for {key}: {absolute_error} > {args.tolerance}"
                )
    if not comparisons:
        raise ValueError(
            "Checkpoint metrics do not contain fixed-evaluation values; use initial.pt or best.pt"
        )
    result = {
        "test": "fresh checkpoint strict resume and fixed-evaluation parity",
        "checkpoint": str(checkpoint_path),
        "checkpoint_version": saved["checkpoint_version"],
        "global_step": trainer.global_step,
        "optimizer_state_parameter_count": len(trainer.optimizer.state),
        "lr_scheduler_last_epoch": trainer.lr_scheduler.last_epoch,
        "tolerance": args.tolerance,
        "comparisons": comparisons,
        "passed": True,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
