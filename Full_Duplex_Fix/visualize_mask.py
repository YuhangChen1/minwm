from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision

from .layout import InterleavedLayout
from .mask import readable_span_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="Full_Duplex_Fix/outputs/mask")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    layout = InterleavedLayout.main()
    text = readable_span_mask(layout)
    (output_dir / "span_mask.txt").write_text(text + "\n", encoding="utf-8")
    mask = layout.span_visibility_matrix().float().unsqueeze(0)
    mask = torch.nn.functional.interpolate(
        mask.unsqueeze(0), size=(780, 780), mode="nearest"
    )[0]
    torchvision.io.write_png(mask.mul(255).byte(), str(output_dir / "span_mask.png"))
    print(text)


if __name__ == "__main__":
    main()
