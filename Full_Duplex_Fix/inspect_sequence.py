from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .layout import InterleavedLayout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="Full_Duplex_Fix/outputs/layout.json")
    args = parser.parse_args()
    layout = InterleavedLayout.main()
    payload = {
        "name": layout.name,
        "num_spans": layout.num_spans,
        "num_noisy_spans": len(layout.noisy_spans),
        "num_clean_spans": len(layout.clean_spans),
        "tokens_per_span": layout.tokens_per_span,
        "sequence_length": layout.sequence_length,
        "labels": layout.labels(),
        "spans": [asdict(span) for span in layout.spans],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
