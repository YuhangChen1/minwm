from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from full_duplex.config import refresh_training_config_hash
from full_duplex.lora import DEFAULT_LORA_TARGETS
from full_duplex.training import FullDuplexTrainer


def _source_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"training_config", "mode", "model", "model_keys", "model_state_format"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise KeyError(f"LoRA source checkpoint is missing keys: {missing}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Warm-start the Full-Duplex task delta and train LoRA only in the physical "
            "last N blocks of the complete Wan Transformer"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The controls are independent:
  --max-steps             target global optimizer step for this LoRA run
  --num-denoising-steps   Euler/Flow updates inside every micro-turn
  --lora-last-blocks      physical final Wan blocks receiving adapters
  --spatial-token-stride  spatial patch-grid sampling interval, not frame stride

LoRA mode always executes all 30 checkpoint blocks. Base Wan weights and the
warm-started Full-Duplex task modules are frozen by default; only A/B matrices
inside the selected final blocks receive gradients.
""",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--warm-start",
        type=Path,
        help="Pre-LoRA Full-Duplex delta; resets optimizer/global step",
    )
    source_group.add_argument(
        "--resume",
        type=Path,
        help="LoRA checkpoint; restores optimizer/global step exactly",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-turns", type=int)
    parser.add_argument("--num-denoising-steps", type=int)
    parser.add_argument(
        "--num-backbone-blocks",
        type=int,
        default=30,
        help="Complete checkpoint depth to execute; current Wan checkpoint has 30",
    )
    parser.add_argument(
        "--spatial-token-stride",
        type=int,
        help="8=28, 4=104, 2=390, 1=1560 tokens per world modality/turn",
    )
    parser.add_argument(
        "--lora-last-blocks",
        type=int,
        help="Number of physical final blocks to adapt; e.g. 4 selects indices 26..29",
    )
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument(
        "--lora-targets",
        nargs="+",
        help="Relative nn.Linear paths inside every selected WanAttentionBlock",
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--attention-pad-to-turns", type=int)
    parser.add_argument(
        "--train-task-modules",
        action="store_true",
        help="Also update Full-Duplex embeddings/heads; default trains only LoRA",
    )
    parser.add_argument(
        "--override-resume-learning-rate",
        action="store_true",
        help="Required when changing LR during an exact LoRA resume",
    )
    args = parser.parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")

    source_path = (args.warm_start or args.resume).resolve()
    checkpoint = _source_checkpoint(source_path)
    config = dict(checkpoint["training_config"])
    mode = checkpoint["mode"]
    del checkpoint
    if mode != "rollout":
        raise ValueError(f"The requested LoRA video experiment requires rollout mode, got {mode}")

    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir.resolve())
    config["max_steps"] = int(args.max_steps)
    if args.num_turns is not None:
        config["rollout_num_turns"] = int(args.num_turns)

    if args.warm_start is not None:
        # The physical last blocks are blocks 26..29 for last_blocks=4. They are
        # reached only when the full 30-layer checkpoint path is executed.
        config["num_backbone_blocks"] = int(args.num_backbone_blocks)
        config["spatial_token_stride"] = int(
            args.spatial_token_stride if args.spatial_token_stride is not None else 8
        )
        config["num_denoising_steps"] = int(
            args.num_denoising_steps if args.num_denoising_steps is not None else 10
        )
        config["attention_pad_to_turns"] = int(
            args.attention_pad_to_turns or config["num_micro_turns"]
        )
        config["train_backbone"] = False
        config["train_base_world_head"] = False
        config["lora_enabled"] = True
        config["lora_last_blocks"] = int(
            args.lora_last_blocks if args.lora_last_blocks is not None else 4
        )
        config["lora_rank"] = int(args.lora_rank if args.lora_rank is not None else 8)
        config["lora_alpha"] = float(
            args.lora_alpha if args.lora_alpha is not None else config["lora_rank"]
        )
        config["lora_dropout"] = float(
            args.lora_dropout if args.lora_dropout is not None else 0.0
        )
        config["lora_targets"] = list(args.lora_targets or DEFAULT_LORA_TARGETS)
        config["lora_train_task_modules"] = bool(args.train_task_modules)
        config["learning_rate"] = float(
            args.learning_rate if args.learning_rate is not None else 1.0e-4
        )
        config["weight_decay"] = float(
            args.weight_decay if args.weight_decay is not None else 0.0
        )
        config["max_grad_norm"] = float(
            args.max_grad_norm if args.max_grad_norm is not None else 1.0
        )
        # In the default "only LoRA" mode there are no trainable world-head or
        # prior parameters, so their optimizer multipliers must be neutral.
        config["world_head_learning_rate_multiplier"] = 1.0
        config["world_prior_learning_rate_multiplier"] = 1.0
        config["lora_learning_rate_multiplier"] = 1.0
        config["gradient_checkpointing"] = True
    else:
        # Resume uses the checkpoint architecture exactly. Optional values are
        # still applied so the ordinary strict resume validator can reject any
        # accidental architecture mismatch rather than ignoring it.
        overrides = {
            "spatial_token_stride": args.spatial_token_stride,
            "num_backbone_blocks": args.num_backbone_blocks,
            "num_denoising_steps": args.num_denoising_steps,
            "lora_last_blocks": args.lora_last_blocks,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_targets": args.lora_targets,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "attention_pad_to_turns": args.attention_pad_to_turns,
        }
        for key, value in overrides.items():
            if value is not None:
                config[key] = value

    for key in (
        "num_denoising_steps",
        "num_backbone_blocks",
        "spatial_token_stride",
        "lora_last_blocks",
        "lora_rank",
    ):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive, got {config[key]}")
    refresh_training_config_hash(config)
    controls = {
        "source": str(source_path),
        "source_mode": "warm_start" if args.warm_start else "resume",
        "target_global_optimizer_step": config["max_steps"],
        "num_micro_turns": config.get("rollout_num_turns", config["num_micro_turns"]),
        "denoising_steps_per_micro_turn": config["num_denoising_steps"],
        "executed_backbone_blocks": config["num_backbone_blocks"],
        "spatial_token_stride": config["spatial_token_stride"],
        "lora_last_blocks": config["lora_last_blocks"],
        "lora_rank": config["lora_rank"],
        "lora_alpha": config["lora_alpha"],
        "lora_targets": config["lora_targets"],
        "train_task_modules": config["lora_train_task_modules"],
        "learning_rate": config["learning_rate"],
    }
    print(f"[LoRA controls] {json.dumps(controls, sort_keys=True)}", flush=True)

    trainer = FullDuplexTrainer(config, mode=mode, run_name=args.run_name)
    trainer.train(
        max_steps=config["max_steps"],
        warm_start=source_path if args.warm_start else None,
        resume=source_path if args.resume else None,
        override_resume_learning_rate=args.override_resume_learning_rate,
    )


if __name__ == "__main__":
    main()
