from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config
from .training import OverfitTrainer
from .wandb_tracking import add_wandb_arguments, wandb_overrides_from_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Full_Duplex_Fix/configs/overfit.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        default="Full_Duplex_Fix/outputs/smallest_000000/optimizer_step_smoke.json",
    )
    add_wandb_arguments(parser)
    args = parser.parse_args()
    wandb_overrides = wandb_overrides_from_args(args)
    wandb_overrides["wandb_job_type"] = "optimizer_step_smoke"
    trainer = OverfitTrainer(
        load_config(args.config),
        device=torch.device(args.device),
        wandb_overrides=wandb_overrides,
    )
    exit_code = 1
    try:
        trainer.start_tracking()
        record = trainer.train_step()
        trainer.log_record("smoke_train_step", record)
        result = {
            "test": "one real full-model AdamW optimizer step without checkpoint serialization",
            "global_step": trainer.global_step,
            "parameter_manifest": trainer.model.parameter_manifest(),
            "optimizer_state_parameter_count": len(trainer.optimizer.state),
            "lr_scheduler_last_epoch": trainer.lr_scheduler.last_epoch,
            "record": record,
        }
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        exit_code = 0
    finally:
        trainer.finish_tracking(exit_code=exit_code)


if __name__ == "__main__":
    main()
