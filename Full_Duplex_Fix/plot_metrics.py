from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics", default="Full_Duplex_Fix/outputs/smallest_000000/metrics.jsonl"
    )
    parser.add_argument(
        "--output", default="Full_Duplex_Fix/outputs/smallest_000000/loss_curve.png"
    )
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in Path(args.metrics).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("The metrics file contains no records")

    import matplotlib.pyplot as plt

    steps = [record["step"] for record in records]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(steps, [record["loss"] for record in records], label="total")
    axis.plot(steps, [record["loss_init"] for record in records], label="init N0")
    axis.plot(
        steps,
        [record["loss_transition"] for record in records],
        label="transitions N1..N19",
    )
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("weighted flow MSE")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
