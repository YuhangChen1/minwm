from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from full_duplex.config import load_config, refresh_training_config_hash
from full_duplex.training import FullDuplexTrainer


def _gradient_norm(loss: torch.Tensor, parameter: torch.Tensor) -> float:
    gradient = torch.autograd.grad(loss, parameter, retain_graph=True, allow_unused=False)[0]
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Gradient audit found NaN/Inf")
    norm = float(gradient.float().norm())
    if norm == 0.0:
        raise AssertionError("Gradient audit found a zero gradient")
    return norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full_duplex/configs/overfit.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("full_duplex/outputs/smallest_000000/gradient_audit.json"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config.update(
        num_backbone_blocks=1,
        spatial_token_stride=8,
        attention_pad_to_turns=0,
        train_backbone=False,
        world_residual_head=True,
    )
    refresh_training_config_hash(config)
    trainer = FullDuplexTrainer(config, mode="single", run_name="gradient_audit_probe")
    trainer.model.clear_step_cache()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = trainer.forward_loss()
    report = {
        "flow_to_special_embedding_grad_norm": _gradient_norm(
            output.flow_loss, trainer.model.special_embedding.weight
        ),
        "state_to_special_embedding_grad_norm": _gradient_norm(
            output.state_loss, trainer.model.special_embedding.weight
        ),
        "flow_to_world_residual_head_grad_norm": _gradient_norm(
            output.flow_loss, trainer.model.world_residual_head.weight
        ),
        "state_to_world_residual_head_grad_norm": _gradient_norm(
            output.state_loss, trainer.model.world_residual_head.weight
        ),
        "camera_to_camera_head_grad_norm": _gradient_norm(
            output.camera_loss, trainer.model.camera_head.output.weight
        ),
        "flow_loss": float(output.flow_loss.detach()),
        "state_loss": float(output.state_loss.detach()),
        "camera_loss": float(output.camera_loss.detach()),
        "real_cache_used": True,
        "num_denoising_steps": config["num_denoising_steps"],
        "finite": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
