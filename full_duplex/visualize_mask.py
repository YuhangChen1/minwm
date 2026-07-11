from __future__ import annotations

import argparse
import json
from pathlib import Path

from full_duplex.tokens import (
    SpecialTokenVocabulary,
    build_attention_mask,
    build_layout,
    render_mask,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("full_duplex/outputs/smallest_000000/mask_visualization.txt"),
    )
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--world-tokens", type=int, default=2)
    args = parser.parse_args()
    vocabulary = SpecialTokenVocabulary(max_time_index=max(31, args.turns - 1))
    layout = build_layout(args.turns, args.world_tokens, 1, vocabulary)
    mask = build_attention_mask(layout)
    span_rows = [
        {
            "turn": span.turn,
            "name": span.name,
            "start": span.start,
            "end": span.end,
            "output_content": span.is_output_content,
            "special": span.is_special,
        }
        for span in layout.spans
    ]
    rendered = (
        "Legend: # visible, . hidden; rows=query, columns=key\n"
        f"shape={tuple(mask.shape)} prediction_tokens={int(layout.prediction_mask.sum())}\n\n"
        + render_mask(mask)
        + "\n\nToken spans:\n"
        + json.dumps(span_rows, indent=2)
        + "\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"[mask visualization] {args.output.resolve()}")


if __name__ == "__main__":
    main()
