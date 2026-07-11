from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from full_duplex.config import load_config
from full_duplex.tokens import SpecialTokenVocabulary, build_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full_duplex/configs/overfit.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("full_duplex/outputs/smallest_000000/full_sequence_layout.json"),
    )
    parser.add_argument("--spatial-token-stride", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    stride = args.spatial_token_stride or config["spatial_token_stride"]
    _, patch_height, patch_width = config["patch_size"]
    patch_rows = config["latent_height"] // patch_height
    patch_columns = config["latent_width"] // patch_width
    world_tokens = math.ceil(patch_rows / stride) * math.ceil(patch_columns / stride)
    vocabulary = SpecialTokenVocabulary(config["max_time_index"])
    layout = build_layout(
        config["num_micro_turns"],
        world_tokens,
        config["num_camera_tokens"],
        vocabulary,
    )
    spans = [
        {
            "turn": span.turn,
            "name": span.name,
            "start": span.start,
            "end": span.end,
            "length": span.length,
            "token_type": span.token_type.name,
            "is_output_content": span.is_output_content,
            "is_special": span.is_special,
        }
        for span in layout.spans
    ]
    turn_boundaries = [
        {
            "turn": turn,
            "start": min(span.start for span in layout.spans if span.turn == turn),
            "end": max(span.end for span in layout.spans if span.turn == turn),
        }
        for turn in range(config["num_micro_turns"])
    ]
    report = {
        "num_turns": config["num_micro_turns"],
        "world_tokens_per_span": world_tokens,
        "camera_tokens_per_span": config["num_camera_tokens"],
        "spatial_token_stride": stride,
        "sequence_length": layout.sequence_length,
        "prediction_token_count": int(layout.prediction_mask.sum()),
        "special_token_count": len(vocabulary),
        "special_token_ids": vocabulary.as_dict(),
        "turn_boundaries": turn_boundaries,
        "spans": spans,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "spans"}, indent=2))
    print(f"[sequence layout] {args.output.resolve()}")


if __name__ == "__main__":
    main()
