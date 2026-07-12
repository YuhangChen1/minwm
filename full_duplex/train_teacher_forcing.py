from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from full_duplex.config import refresh_training_config_hash
from full_duplex.teacher_forcing_training import (
    TEACHER_FORCED_REGIME,
    TeacherForcedTransitionTrainer,
)


def _source_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "training_config",
        "mode",
        "model",
        "model_keys",
        "model_state_format",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise KeyError(f"Teacher-forcing source checkpoint is missing keys: {missing}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train exact previous-ground-truth -> next-ground-truth Full-Duplex "
            "latent transitions without a cross-turn autograd graph"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The three controls below have independent meanings:
  --max-steps             optimizer updates over the complete transition set
  --num-denoising-steps   differentiable Flow/Euler updates inside each transition
  --blocks                leading pretrained Wan Transformer layers executed per call
  --spatial-token-stride  spatial patch-grid interval, not a temporal/video stride

For the cached 60x104 latent and 2x2 patches, stride 8/4/2/1 selects
28/104/390/1560 world tokens per modality per turn. Smaller stride is denser.

Turn 0 receives an all-zero world state. Turn t>0 receives cached ground-truth
state[t] and predicts cached state[t+1]. Each transition is backpropagated
immediately (loss/num_turns), so one optimizer update is still the mean over all
turns while no 19-turn autograd graph is retained. This path never enables LoRA.
""",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--warm-start",
        type=Path,
        help="Existing non-LoRA Full-Duplex task delta; resets optimizer/global step",
    )
    source_group.add_argument(
        "--resume",
        type=Path,
        help="Teacher-forced checkpoint; restores optimizer/global step exactly",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--max-steps",
        required=True,
        type=int,
        help="Target global optimizer step; not the number of denoising updates",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-turns", type=int)
    parser.add_argument(
        "--num-denoising-steps",
        type=int,
        help="Flow/Euler steps inside each micro-turn; defaults to 10 on warm start",
    )
    parser.add_argument(
        "--blocks",
        "--num-backbone-blocks",
        dest="blocks",
        type=int,
        help=(
            "Leading pretrained Wan Transformer layers executed per model call; "
            "the checkpoint contains 30. Defaults to 4 on warm start"
        ),
    )
    parser.add_argument(
        "--spatial-token-stride",
        type=int,
        help=(
            "Spatial sampling interval on the 30x52 patch grid: "
            "8=28, 4=104, 2=390, 1=1560 tokens; not frame/action stride"
        ),
    )
    parser.add_argument("--max-history-turns", type=int)
    parser.add_argument("--attention-pad-to-turns", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--world-head-learning-rate-multiplier", type=float)
    parser.add_argument("--world-prior-learning-rate-multiplier", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Recompute Transformer block activations during backward to save memory. "
            "Use --no-gradient-checkpointing to trade HBM for speed"
        ),
    )
    parser.add_argument(
        "--checkpoint-blocks",
        type=int,
        help=(
            "Checkpoint only the leading N executed Transformer blocks. For 30 layers, "
            "N=30 minimizes HBM, N=0 stores all activations, and intermediate values "
            "trade HBM for speed without changing forward values"
        ),
    )
    parser.add_argument(
        "--teacher-force-camera",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Also feed exact previous camera ground truth. Default is false so the "
            "requested behavior change applies only to world-state input"
        ),
    )
    parser.add_argument(
        "--override-resume-learning-rate",
        action="store_true",
        help="Explicitly replace restored optimizer LR while preserving moments",
    )
    args = parser.parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.checkpoint_blocks is not None and args.checkpoint_blocks < 0:
        raise ValueError("--checkpoint-blocks must be non-negative")
    if args.checkpoint_blocks is not None and args.gradient_checkpointing is not None:
        raise ValueError(
            "Use either --checkpoint-blocks or --[no-]gradient-checkpointing, not both"
        )

    source_path = (args.warm_start or args.resume).resolve()
    checkpoint = _source_checkpoint(source_path)
    config = dict(checkpoint["training_config"])
    mode = checkpoint["mode"]
    source_format = checkpoint["model_state_format"]
    del checkpoint
    if mode != "rollout":
        raise ValueError(f"Teacher-forced video transitions require rollout mode, got {mode}")

    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir.resolve())
    config["max_steps"] = int(args.max_steps)
    if args.num_turns is not None:
        config["rollout_num_turns"] = int(args.num_turns)

    if args.warm_start is not None:
        if source_format != "trainable_delta_over_strict_base":
            raise ValueError(
                "--warm-start requires a non-LoRA task delta, got " + source_format
            )
        # Only the rollout input/gradient regime changes. Keep the learned task
        # modules and loss/LR settings from the source checkpoint, execute more
        # frozen pretrained blocks as memory permits, and explicitly disable LoRA.
        config["training_regime"] = TEACHER_FORCED_REGIME
        config["teacher_forced_world_inputs"] = True
        config["teacher_force_camera"] = bool(args.teacher_force_camera or False)
        config["sequential_turn_backward"] = True
        config["teacher_forcing_ratio"] = 1.0
        config["detach_between_turns"] = True
        config["num_backbone_blocks"] = int(args.blocks if args.blocks is not None else 4)
        config["spatial_token_stride"] = int(
            args.spatial_token_stride if args.spatial_token_stride is not None else 8
        )
        config["num_denoising_steps"] = int(
            args.num_denoising_steps if args.num_denoising_steps is not None else 10
        )
        config["attention_pad_to_turns"] = int(
            args.attention_pad_to_turns
            if args.attention_pad_to_turns is not None
            else config["num_micro_turns"]
        )
        config["train_backbone"] = False
        config["train_base_world_head"] = False
        config["lora_enabled"] = False
        config["lora_train_task_modules"] = False
        config["lora_learning_rate_multiplier"] = 1.0
        config["gradient_checkpointing"] = (
            bool(args.gradient_checkpointing)
            if args.gradient_checkpointing is not None
            else True
        )
        config["gradient_checkpointing_blocks"] = (
            int(args.checkpoint_blocks)
            if args.checkpoint_blocks is not None
            else -1
        )
        if args.checkpoint_blocks is not None:
            config["gradient_checkpointing"] = args.checkpoint_blocks > 0
    else:
        if config.get("training_regime") != TEACHER_FORCED_REGIME:
            raise ValueError(
                "--resume requires a checkpoint produced by train_teacher_forcing.py"
            )
        overrides = {
            "num_backbone_blocks": args.blocks,
            "spatial_token_stride": args.spatial_token_stride,
            "num_denoising_steps": args.num_denoising_steps,
            "max_history_turns": args.max_history_turns,
            "attention_pad_to_turns": args.attention_pad_to_turns,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "world_head_learning_rate_multiplier": (
                args.world_head_learning_rate_multiplier
            ),
            "world_prior_learning_rate_multiplier": (
                args.world_prior_learning_rate_multiplier
            ),
            "max_grad_norm": args.max_grad_norm,
            "teacher_force_camera": args.teacher_force_camera,
            "gradient_checkpointing": args.gradient_checkpointing,
            "gradient_checkpointing_blocks": args.checkpoint_blocks,
        }
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
        if args.checkpoint_blocks is not None:
            config["gradient_checkpointing"] = args.checkpoint_blocks > 0
        elif args.gradient_checkpointing is not None:
            config["gradient_checkpointing_blocks"] = -1 if args.gradient_checkpointing else 0

    common_overrides = {
        "max_history_turns": args.max_history_turns,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "world_head_learning_rate_multiplier": args.world_head_learning_rate_multiplier,
        "world_prior_learning_rate_multiplier": args.world_prior_learning_rate_multiplier,
        "max_grad_norm": args.max_grad_norm,
    }
    for key, value in common_overrides.items():
        if value is not None:
            config[key] = value

    if config.get("lora_enabled", False):
        raise ValueError("Teacher-forced transition training must not enable LoRA")
    checkpoint_blocks = int(config.get("gradient_checkpointing_blocks", -1))
    if checkpoint_blocks < 0:
        checkpoint_blocks = (
            int(config["num_backbone_blocks"])
            if config["gradient_checkpointing"]
            else 0
        )
    if not 0 <= checkpoint_blocks <= int(config["num_backbone_blocks"]):
        raise ValueError("checkpointed block count must fit --blocks")
    if not config["gradient_checkpointing"] and checkpoint_blocks != 0:
        raise ValueError("Disabled gradient checkpointing requires zero checkpointed blocks")
    for key in (
        "max_steps",
        "num_denoising_steps",
        "num_backbone_blocks",
        "spatial_token_stride",
    ):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive, got {config[key]}")
    num_turns = int(config.get("rollout_num_turns", config["num_micro_turns"]))
    if not 1 <= num_turns <= int(config["num_micro_turns"]):
        raise ValueError("--num-turns must fit the cached transition count")

    refresh_training_config_hash(config)
    controls = {
        "source": str(source_path),
        "source_mode": "warm_start" if args.warm_start else "resume",
        "training_regime": config["training_regime"],
        "target_global_optimizer_step": config["max_steps"],
        "num_transitions_per_optimizer_step": num_turns,
        "denoising_steps_per_transition": config["num_denoising_steps"],
        "executed_backbone_blocks": config["num_backbone_blocks"],
        "spatial_token_stride": config["spatial_token_stride"],
        "teacher_forced_world_inputs": True,
        "teacher_forced_camera_inputs": config["teacher_force_camera"],
        "cross_turn_bptt": False,
        "gradient_checkpointing": config["gradient_checkpointing"],
        "gradient_checkpointing_blocks": checkpoint_blocks,
        "lora_enabled": False,
    }
    print(f"[teacher-forcing controls] {json.dumps(controls, sort_keys=True)}", flush=True)

    trainer = TeacherForcedTransitionTrainer(config, mode=mode, run_name=args.run_name)
    trainer.train(
        max_steps=config["max_steps"],
        warm_start=source_path if args.warm_start else None,
        resume=source_path if args.resume else None,
        override_resume_learning_rate=args.override_resume_learning_rate,
    )


if __name__ == "__main__":
    main()
