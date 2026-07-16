from __future__ import annotations

import argparse
import json

import torch

from .checkpoint import load_strict_generator
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Full_Duplex_Fix/configs/overfit.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    generator, audit = load_strict_generator(
        config,
        device=args.device,
        dtype=torch.float32,
    )
    audit["trainable_parameter_count"] = sum(
        parameter.numel() for parameter in generator.parameters() if parameter.requires_grad
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
