from __future__ import annotations

import argparse
import json

from full_duplex.config import load_config, refresh_training_config_hash
from full_duplex.training import FullDuplexTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-Duplex Wan latent overfit worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Do not confuse the two kinds of steps:
  --max-steps             target global optimizer step (fresh run: update count)
  --num-denoising-steps   number of differentiable Euler updates per micro-turn

--blocks controls Transformer depth. --spatial-token-stride controls only
spatial patch density; it does not change RGB frames, actions, or temporal turns.
""",
    )
    parser.add_argument("--config", default="full_duplex/configs/overfit.yaml")
    parser.add_argument("--mode", choices=("single", "rollout"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--max-steps",
        type=int,
        help=(
            "Target global optimizer step; fresh runs perform this many updates and resumed "
            "runs continue only until this value. Not denoising steps"
        ),
    )
    parser.add_argument("--resume")
    parser.add_argument("--num-turns", type=int)
    parser.add_argument(
        "--num-denoising-steps",
        type=int,
        help="Flow/Euler updates inside every micro-turn; current trained baseline uses 10",
    )
    parser.add_argument(
        "--blocks",
        "--num-backbone-blocks",
        dest="blocks",
        type=int,
        help="Leading pretrained Wan Transformer blocks to execute; valid range is 1..30",
    )
    parser.add_argument(
        "--spatial-token-stride",
        type=int,
        help=(
            "Spatial sampling interval on the 30x52 patch grid: "
            "8=28, 4=104, 2=390, 1=1560 tokens; not temporal stride"
        ),
    )
    parser.add_argument("--max-history-turns", type=int)
    parser.add_argument("--attention-pad-to-turns", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--world-head-learning-rate-multiplier", type=float)
    parser.add_argument("--world-prior-learning-rate-multiplier", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument(
        "--override-resume-learning-rate",
        action="store_true",
        help="Explicitly replace the restored optimizer LR while preserving its moments",
    )
    parser.add_argument("--world-residual-head", action="store_true")
    parser.add_argument("--world-time-space-prior", action="store_true")
    parser.add_argument("--train-base-world-head", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    overrides = {
        "num_denoising_steps": args.num_denoising_steps,
        "num_backbone_blocks": args.blocks,
        "spatial_token_stride": args.spatial_token_stride,
        "max_history_turns": args.max_history_turns,
        "attention_pad_to_turns": args.attention_pad_to_turns,
        "learning_rate": args.learning_rate,
        "world_head_learning_rate_multiplier": args.world_head_learning_rate_multiplier,
        "world_prior_learning_rate_multiplier": args.world_prior_learning_rate_multiplier,
        "max_grad_norm": args.max_grad_norm,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.num_turns is not None:
        config["rollout_num_turns"] = args.num_turns
    if args.freeze_backbone:
        config["train_backbone"] = False
    if args.disable_gradient_checkpointing:
        config["gradient_checkpointing"] = False
    if args.world_residual_head:
        config["world_residual_head"] = True
    if args.world_time_space_prior:
        config["world_time_space_prior"] = True
    if args.train_base_world_head:
        config["train_base_world_head"] = True
    if args.max_steps is not None:
        # Persist the effective CLI value into manifests and checkpoints. This
        # is the target global optimizer step (not an extra-resume count and
        # not the per-turn denoising-step count).
        config["max_steps"] = args.max_steps
    for key in (
        "max_steps",
        "num_denoising_steps",
        "num_backbone_blocks",
        "spatial_token_stride",
    ):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive, got {config[key]}")
    refresh_training_config_hash(config)
    controls = {
        "target_global_optimizer_step": int(config["max_steps"]),
        "denoising_steps_per_micro_turn": int(config["num_denoising_steps"]),
        "backbone_blocks_per_model_call": int(config["num_backbone_blocks"]),
        "spatial_patch_grid_stride": int(config["spatial_token_stride"]),
    }
    print(f"[training controls] {json.dumps(controls, sort_keys=True)}", flush=True)
    trainer = FullDuplexTrainer(config, args.mode, args.run_name)
    trainer.train(
        max_steps=config["max_steps"],
        resume=args.resume,
        override_resume_learning_rate=args.override_resume_learning_rate,
    )


if __name__ == "__main__":
    main()
