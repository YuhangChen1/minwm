from __future__ import annotations

import argparse

from full_duplex.config import load_config, refresh_training_config_hash
from full_duplex.training import FullDuplexTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full_duplex/configs/overfit.yaml")
    parser.add_argument("--mode", choices=("single", "rollout"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--num-turns", type=int)
    parser.add_argument("--num-denoising-steps", type=int)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--spatial-token-stride", type=int)
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
    refresh_training_config_hash(config)
    max_steps = args.max_steps if args.max_steps is not None else config["max_steps"]
    trainer = FullDuplexTrainer(config, args.mode, args.run_name)
    trainer.train(
        max_steps=max_steps,
        resume=args.resume,
        override_resume_learning_rate=args.override_resume_learning_rate,
    )


if __name__ == "__main__":
    main()
