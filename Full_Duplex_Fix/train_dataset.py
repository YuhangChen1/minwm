from __future__ import annotations

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from .config import load_config
from .dataset_training import MultiSampleTrainer
from .wandb_tracking import add_wandb_arguments, wandb_overrides_from_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="Full_Duplex_Fix/configs/train_1000.yaml"
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", default=None)
    add_wandb_arguments(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.get("training_mode") != "multi_sample":
        raise ValueError("train_dataset requires training_mode=multi_sample")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    try:
        trainer = MultiSampleTrainer(
            config,
            wandb_overrides=wandb_overrides_from_args(args),
        )
        if args.resume:
            trainer.load_checkpoint(args.resume)
        max_steps = int(
            args.max_steps if args.max_steps is not None else config["max_steps"]
        )
        trainer.train(max_steps)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
