from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch

from .config import load_config
from .debug.tracer import DebugTracer, debug_scope
from .preencode import preencode
from .training import OverfitTrainer
from .wandb_tracking import add_wandb_arguments, wandb_overrides_from_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Full_Duplex_Fix/configs/overfit.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--skip-preencode", action="store_true")
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Enable JSON debug instrumentation; the interactive UI uses this mode internally.",
    )
    add_wandb_arguments(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    debug_tracer = (
        DebugTracer(
            log_path=Path(config["output_dir"]) / "debug_events.jsonl",
            mode="auto",
        )
        if args.debug_mode
        else None
    )
    context = debug_scope(debug_tracer) if debug_tracer is not None else nullcontext()
    with context:
        if not args.skip_preencode:
            preencode(config, device=args.device)
        trainer = OverfitTrainer(
            config,
            device=torch.device(args.device),
            wandb_overrides=wandb_overrides_from_args(args),
            debug_mode=args.debug_mode,
            debug_tracer=debug_tracer,
        )
        if args.resume:
            trainer.load_checkpoint(args.resume)
        max_steps = int(args.max_steps if args.max_steps is not None else config["max_steps"])
        trainer.train(max_steps)


if __name__ == "__main__":
    main()
