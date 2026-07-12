from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from full_duplex.config import load_config


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronous, foreground controller for Full-Duplex overfit runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The resource controls are independent:
  --max-steps             target global optimizer step (fresh run: update count)
  --num-denoising-steps   Flow/Euler updates inside every micro-turn
  --blocks                pretrained Wan Transformer layers executed per model call
  --spatial-token-stride  spatial patch-grid sampling interval (not video-frame stride)

For the cached 60x104 latent and 2x2 patches, stride 8/4/2/1 selects
28/104/390/1560 world tokens per modality per turn, respectively.
""",
    )
    parser.add_argument("--config", default="full_duplex/configs/overfit.yaml")
    parser.add_argument("--mode", choices=("single", "rollout"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--max-steps",
        type=int,
        required=True,
        help=(
            "Target global optimizer step; on a fresh run this is the update count, while "
            "resume continues only until this value. Not denoising iterations"
        ),
    )
    parser.add_argument("--resume")
    parser.add_argument("--num-turns", type=int)
    parser.add_argument(
        "--num-denoising-steps",
        type=int,
        help=(
            "Differentiable Flow/Euler steps inside each micro-turn (10 means "
            "19 turns execute 190 growing-history model calls per rollout)"
        ),
    )
    parser.add_argument(
        "--blocks",
        "--num-backbone-blocks",
        dest="blocks",
        type=int,
        help=(
            "Number of leading pretrained Wan Transformer blocks executed per model call; "
            "the current checkpoint contains 30"
        ),
    )
    parser.add_argument(
        "--spatial-token-stride",
        type=int,
        help=(
            "Sampling interval on the 30x52 latent patch grid; smaller is denser and more "
            "expensive (8=28, 4=104, 2=390, 1=1560 tokens). Not temporal stride"
        ),
    )
    parser.add_argument("--max-history-turns", type=int)
    parser.add_argument("--attention-pad-to-turns", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--world-head-learning-rate-multiplier", type=float)
    parser.add_argument("--world-prior-learning-rate-multiplier", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--override-resume-learning-rate", action="store_true")
    parser.add_argument("--world-residual-head", action="store_true")
    parser.add_argument("--world-time-space-prior", action="store_true")
    parser.add_argument("--train-base-world-head", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(config["output_dir"]) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_log = run_dir / "controller_train.log"
    status_path = run_dir / "controller_status.json"
    summary_path = run_dir / "controller_summary.json"

    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("train_overfit.py")),
        "--config",
        args.config,
        "--mode",
        args.mode,
        "--run-name",
        args.run_name,
        "--max-steps",
        str(args.max_steps),
    ]
    optional_values = (
        ("--resume", args.resume),
        ("--num-turns", args.num_turns),
        ("--num-denoising-steps", args.num_denoising_steps),
        ("--blocks", args.blocks),
        ("--spatial-token-stride", args.spatial_token_stride),
        ("--max-history-turns", args.max_history_turns),
        ("--attention-pad-to-turns", args.attention_pad_to_turns),
        ("--learning-rate", args.learning_rate),
        (
            "--world-head-learning-rate-multiplier",
            args.world_head_learning_rate_multiplier,
        ),
        (
            "--world-prior-learning-rate-multiplier",
            args.world_prior_learning_rate_multiplier,
        ),
        ("--max-grad-norm", args.max_grad_norm),
    )
    for flag, value in optional_values:
        if value is not None:
            command.extend((flag, str(value)))
    if args.freeze_backbone:
        command.append("--freeze-backbone")
    if args.world_residual_head:
        command.append("--world-residual-head")
    if args.world_time_space_prior:
        command.append("--world-time-space-prior")
    if args.train_base_world_head:
        command.append("--train-base-world-head")
    if args.override_resume_learning_rate:
        command.append("--override-resume-learning-rate")

    root = Path(config["project_root"])
    env = os.environ.copy()
    python_paths = (root, root / "Wan21", root / "shared")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in python_paths), *([existing] if existing else [])]
    )
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    started = time.time()
    status: dict[str, Any] = {
        "state": "starting",
        "command": shlex.join(command),
        "pid": None,
        "started_unix": started,
        "last_metric": None,
        "metric_count": 0,
    }
    _atomic_json(status, status_path)
    return_code: int | None = None
    process: subprocess.Popen[str] | None = None
    try:
        with raw_log.open("a", encoding="utf-8", buffering=1) as log_handle:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            status.update(state="running", pid=process.pid)
            _atomic_json(status, status_path)
            assert process.stdout is not None
            for line in process.stdout:
                log_handle.write(line)
                if line.startswith("[train] "):
                    metric = json.loads(line[len("[train] ") :])
                    compact = {
                        key: metric[key]
                        for key in (
                            "step",
                            "total_loss",
                            "flow_loss",
                            "state_loss",
                            "camera_loss",
                            "gradient_norm",
                            "peak_gpu_memory_gib",
                            "elapsed_seconds",
                        )
                    }
                    print(f"[train] {json.dumps(compact, sort_keys=True)}", flush=True)
                    status["last_metric"] = metric
                    status["metric_count"] += 1
                    status["last_update_unix"] = time.time()
                    _atomic_json(status, status_path)
                else:
                    print(line, end="", flush=True)
            return_code = process.wait()
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        status.update(
            state="controller_exception",
            exception_type=type(error).__name__,
            exception=str(error),
            ended_unix=time.time(),
        )
        _atomic_json(status, status_path)
        raise

    status.update(
        state="completed" if return_code == 0 else "failed",
        return_code=return_code,
        ended_unix=time.time(),
        elapsed_seconds=time.time() - started,
    )
    _atomic_json(status, status_path)
    worker_summary_path = run_dir / "summary.json"
    controller_summary = dict(status)
    if worker_summary_path.exists():
        controller_summary["worker_summary"] = json.loads(
            worker_summary_path.read_text(encoding="utf-8")
        )
    _atomic_json(controller_summary, summary_path)
    print(f"[controller summary] {json.dumps(controller_summary, sort_keys=True)}")
    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
