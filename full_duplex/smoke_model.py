from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from full_duplex.camera import camera_loss
from full_duplex.config import load_config
from full_duplex.model import DuplexTurn, FullDuplexWanModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full_duplex/configs/overfit.yaml")
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--spatial-token-stride", type=int)
    parser.add_argument("--disable-prope", action="store_true")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.blocks is not None:
        config["num_backbone_blocks"] = args.blocks
    if args.spatial_token_stride is not None:
        config["spatial_token_stride"] = args.spatial_token_stride
    if args.disable_prope:
        config["use_prope"] = False

    cache = torch.load(
        Path(config["cache_path"]) / "tensors.pt", map_location="cpu", weights_only=True
    )
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model = FullDuplexWanModel.from_checkpoint(config)
    parameter_counts = model.configure_trainable_parameters(config["train_backbone"])
    model = model.to(device=device, dtype=dtype)
    model.train(args.backward)

    target = cache["world_state_latents"][1:2].unsqueeze(0).to(device=device, dtype=dtype)
    world_input = torch.zeros_like(target)
    camera_input = cache["camera"][0:1].to(device=device, dtype=dtype)
    camera_target = cache["camera"][1:2].to(device=device, dtype=dtype)
    action = cache["action_ids"][0:1].to(device)
    generator = torch.Generator(device="cpu").manual_seed(config["fixed_noise_seed"])
    noise = torch.randn(target.shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
    sigma = torch.tensor(0.75, device=device, dtype=dtype)
    noisy = (1 - sigma) * target + sigma * noise
    turn = DuplexTurn(0, world_input, camera_input, action, noisy)
    prompt = cache["prompt_embedding"].unsqueeze(0).to(device=device, dtype=dtype)
    prompt_mask = cache["prompt_attention_mask"].unsqueeze(0).to(device)
    timestep = (sigma.float() * config["num_train_timesteps"]).reshape(1, 1)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    context = torch.enable_grad() if args.backward else torch.inference_mode()
    with context, torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
        output = model([turn], prompt, prompt_mask, timestep)
        flow_target = noise - target
        flow_loss = F.mse_loss(output.flow.float(), flow_target.float())
        state_prediction = noisy - sigma * output.flow
        state_loss = F.mse_loss(state_prediction.float(), target.float())
        camera_losses = camera_loss(
            output.camera,
            camera_target,
            config["lambda_translation"],
            config["lambda_rotation"],
            config["lambda_intrinsics"],
        )
        loss = flow_loss + state_loss + camera_losses.total
    gradient = None
    if args.backward:
        loss.backward()
        finite_gradients = []
        gradient_norm_sq = torch.zeros((), device=device)
        nonzero = 0
        for parameter in model.parameters():
            if parameter.grad is not None:
                finite_gradients.append(bool(torch.isfinite(parameter.grad).all()))
                gradient_norm_sq += parameter.grad.float().pow(2).sum()
                nonzero += int(torch.count_nonzero(parameter.grad).item() > 0)
        gradient = {
            "all_finite": all(finite_gradients),
            "global_norm": float(gradient_norm_sq.sqrt()),
            "parameters_with_nonzero_grad": nonzero,
        }
    torch.cuda.synchronize()
    report = {
        "load_report": model.load_report,
        "parameter_counts": parameter_counts,
        "special_token_count": len(model.vocabulary),
        "special_token_ids": model.vocabulary.as_dict(),
        "new_parameter_names": model.new_parameter_names(),
        "shape_log": model.last_shape_log,
        "sequence_length": output.sequence_length,
        "flow_finite": bool(torch.isfinite(output.flow).all()),
        "camera_finite": bool(torch.isfinite(output.camera).all()),
        "hidden_finite": output.hidden_is_finite,
        "flow_loss": float(flow_loss.detach()),
        "state_loss": float(state_loss.detach()),
        "camera_loss": float(camera_losses.total.detach()),
        "total_loss": float(loss.detach()),
        "gradient": gradient,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "num_backbone_blocks": config["num_backbone_blocks"],
        "spatial_token_stride": config["spatial_token_stride"],
        "use_prope": config["use_prope"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
