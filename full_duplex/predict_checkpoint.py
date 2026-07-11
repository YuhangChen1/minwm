from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from full_duplex.training import FullDuplexTrainer, _atomic_torch_save


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a deterministic latent-space rollout from a saved Full-Duplex checkpoint"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(checkpoint["training_config"])
    mode = checkpoint["mode"]
    run_name = f"checkpoint_prediction_step_{checkpoint['global_step']:06d}"
    trainer = FullDuplexTrainer(config, mode=mode, run_name=run_name)
    trainer.load_checkpoint(args.checkpoint)
    trainer.model.clear_step_cache()
    trainer.model.eval()
    with torch.inference_mode(), torch.autocast(
        "cuda",
        dtype=trainer.compute_dtype,
        enabled=trainer.compute_dtype == torch.bfloat16,
    ):
        output = trainer.forward_loss()
    payload = {
        "states": torch.cat([value.cpu() for value in output.predictions], dim=1),
        "cameras": torch.cat([value.cpu() for value in output.camera_predictions], dim=0),
        "target_states": trainer.world_states[1 : len(output.predictions) + 1].cpu(),
        "target_cameras": trainer.cameras[1 : len(output.camera_predictions) + 1].cpu(),
    }
    _atomic_torch_save(payload, args.output)
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_global_step": checkpoint["global_step"],
        "checkpoint_best_step": checkpoint["best_step"],
        "fixed_noise_sha256": trainer.fixed_noise_sha256,
        "num_turns": len(output.predictions),
        "sequence_length_min": min(output.sequence_lengths),
        "sequence_length_max": max(output.sequence_lengths),
        "states_shape": list(payload["states"].shape),
        "cameras_shape": list(payload["cameras"].shape),
        "finite": all(torch.isfinite(value).all().item() for value in payload.values()),
        "rgb_decoder_used": False,
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
