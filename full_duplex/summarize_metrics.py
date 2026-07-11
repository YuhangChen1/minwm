from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


AGGREGATE_FIELDS = (
    "step",
    "total_loss",
    "flow_loss",
    "state_loss",
    "camera_loss",
    "translation_loss",
    "rotation_loss",
    "intrinsics_loss",
    "gradient_norm",
    "parameter_norm",
    "learning_rate",
    "peak_gpu_memory_gib",
    "elapsed_seconds",
    "early_turn_future_gradient_norm",
)


def _load(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"No metrics in {path}")
    steps = [row["step"] for row in rows]
    if steps != sorted(set(steps)):
        raise ValueError(f"Metrics steps are not unique and increasing: {steps}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    rows = _load(args.metrics)
    output_dir = args.output_dir or args.metrics.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_csv = output_dir / "loss_history.csv"
    with aggregate_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in AGGREGATE_FIELDS})

    per_turn_csv = output_dir / "per_turn_loss_history.csv"
    per_turn_fields = (
        "step",
        "turn",
        "flow_loss",
        "state_loss",
        "camera_loss",
        "translation_loss",
        "rotation_loss",
        "intrinsics_loss",
    )
    with per_turn_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_turn_fields)
        writer.writeheader()
        for row in rows:
            for turn in row["per_turn"]:
                writer.writerow({"step": row["step"], **turn})

    curve_path = output_dir / "loss_curve.png"
    figure, left_axis = plt.subplots(figsize=(9, 5), dpi=160)
    steps = [row["step"] for row in rows]
    for field, label in (
        ("total_loss", "total"),
        ("flow_loss", "flow"),
        ("state_loss", "state"),
    ):
        left_axis.plot(steps, [row[field] for row in rows], label=label, linewidth=1.8)
    left_axis.set_xlabel("optimizer step")
    left_axis.set_ylabel("world / total loss")
    left_axis.grid(alpha=0.25)
    right_axis = left_axis.twinx()
    right_axis.plot(
        steps,
        [row["camera_loss"] for row in rows],
        label="camera",
        color="tab:red",
        linewidth=1.4,
        alpha=0.8,
    )
    right_axis.set_ylabel("camera loss")
    lines = left_axis.lines + right_axis.lines
    left_axis.legend(lines, [line.get_label() for line in lines], loc="upper right")
    figure.tight_layout()
    figure.savefig(curve_path)
    plt.close(figure)

    losses = [row["total_loss"] for row in rows]
    decreasing_pairs = sum(right < left for left, right in zip(losses, losses[1:]))
    report = {
        "metrics_path": str(args.metrics.resolve()),
        "num_logged_steps": len(rows),
        "first_step": rows[0]["step"],
        "last_step": rows[-1]["step"],
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "minimum_step": rows[losses.index(min(losses))]["step"],
        "absolute_loss_decrease": losses[0] - losses[-1],
        "relative_loss_decrease": (losses[0] - losses[-1]) / losses[0],
        "fraction_of_decreasing_transitions": (
            decreasing_pairs / (len(losses) - 1) if len(losses) > 1 else None
        ),
        "maximum_peak_gpu_memory_gib": max(row["peak_gpu_memory_gib"] for row in rows),
        "mean_step_seconds": sum(row["elapsed_seconds"] for row in rows) / len(rows),
        "first_early_turn_future_gradient_norm": rows[0].get(
            "early_turn_future_gradient_norm"
        ),
        "aggregate_csv": str(aggregate_csv.resolve()),
        "per_turn_csv": str(per_turn_csv.resolve()),
        "loss_curve": str(curve_path.resolve()),
    }
    report_path = output_dir / "loss_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
