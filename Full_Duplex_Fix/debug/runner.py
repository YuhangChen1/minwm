from __future__ import annotations

import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..config import load_config
from ..training import OverfitTrainer
from .tracer import DebugRunStopped, DebugTracer, debug_scope, trace_event


DEBUG_DEVICE = torch.device("cuda:2")
DEBUG_STEPS = 5


class DebugRunManager:
    def __init__(
        self,
        *,
        config_path: str | Path = "Full_Duplex_Fix/configs/overfit.yaml",
        runs_root: str | Path = "Full_Duplex_Fix/debug/runs",
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.runs_root = Path(runs_root).resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._tracer: DebugTracer | None = None
        self._session_id: str | None = None
        self._output_dir: Path | None = None

    @property
    def tracer(self) -> DebugTracer | None:
        with self._lock:
            return self._tracer

    def state(self) -> dict[str, Any]:
        with self._lock:
            tracer = self._tracer
            thread = self._thread
            session_id = self._session_id
            output_dir = self._output_dir
        snapshot = tracer.snapshot() if tracer is not None else {
            "status": "not_started",
            "message": "",
            "mode": "step",
            "step": 0,
            "total_steps": DEBUG_STEPS,
            "event_count": 0,
            "current_event_id": None,
            "stop_requested": False,
            "runtime_phase": "forward",
        }
        snapshot.update(
            {
                "session_id": session_id,
                "output_dir": str(output_dir) if output_dir is not None else None,
                "device": str(DEBUG_DEVICE),
                "thread_alive": bool(thread and thread.is_alive()),
                "fixed_steps": DEBUG_STEPS,
            }
        )
        return snapshot

    def events_after(self, event_id: int, limit: int = 200) -> list[dict[str, Any]]:
        tracer = self.tracer
        return [] if tracer is None else tracer.events_after(event_id, limit)

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._tracer is not None:
                return self.state_unlocked()
            now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self._session_id = f"{now}_{uuid.uuid4().hex[:8]}"
            self._output_dir = self.runs_root / self._session_id
            self._output_dir.mkdir(parents=True, exist_ok=False)
            self._tracer = DebugTracer(
                log_path=self._output_dir / "events.jsonl",
                mode="step",
                synchronize_cuda=True,
            )
            self._tracer.set_status("starting", "Training thread is starting")
            self._thread = threading.Thread(
                target=self._run,
                name=f"full-duplex-debug-{self._session_id}",
                daemon=True,
            )
            self._thread.start()
            return self.state_unlocked()

    def state_unlocked(self) -> dict[str, Any]:
        tracer = self._tracer
        snapshot = tracer.snapshot() if tracer is not None else {}
        snapshot.update(
            {
                "session_id": self._session_id,
                "output_dir": str(self._output_dir) if self._output_dir is not None else None,
                "device": str(DEBUG_DEVICE),
                "thread_alive": bool(self._thread and self._thread.is_alive()),
                "fixed_steps": DEBUG_STEPS,
            }
        )
        return snapshot

    def control(self, action: str, *, delay_ms: int | None = None) -> dict[str, Any]:
        tracer = self.tracer
        if tracer is None:
            raise RuntimeError("Debug training has not started")
        if action == "next":
            tracer.continue_one()
        elif action == "auto":
            tracer.run_auto(delay_ms)
        elif action == "pause":
            tracer.pause_at_next()
        elif action == "stop":
            tracer.request_stop()
        else:
            raise ValueError(f"Unsupported control action: {action}")
        return self.state()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Stop or finish the current debug run before reset")
            if self._tracer is not None:
                self._tracer.close()
            self._thread = None
            self._tracer = None
            self._session_id = None
            self._output_dir = None
        return self.state()

    def _run(self) -> None:
        tracer = self.tracer
        if tracer is None:
            return
        trainer: OverfitTrainer | None = None
        cuda_initialized = False
        exit_status = "error"
        try:
            with debug_scope(tracer):
                tracer.set_step(0, DEBUG_STEPS)
                trace_event(
                    "session",
                    "session.debug_runner_started",
                    details={
                        "device": str(DEBUG_DEVICE),
                        "fixed_steps": DEBUG_STEPS,
                        "config_path": str(self.config_path),
                        "output_dir": str(self._output_dir),
                        "checkpoint_serialization": False,
                        "wandb": False,
                    },
                )
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available")
                if torch.cuda.device_count() < 3:
                    raise RuntimeError(
                        f"cuda:2 requires at least three visible GPUs; found {torch.cuda.device_count()}"
                    )
                torch.cuda.set_device(DEBUG_DEVICE)
                cuda_initialized = True
                properties = torch.cuda.get_device_properties(DEBUG_DEVICE)
                free_bytes, total_bytes = torch.cuda.mem_get_info(DEBUG_DEVICE)
                trace_event(
                    "gpu",
                    "session.cuda_2_ready",
                    details={
                        "name": properties.name,
                        "total_gib": total_bytes / 2**30,
                        "free_gib": free_bytes / 2**30,
                        "compute_capability": f"{properties.major}.{properties.minor}",
                    },
                )

                config = load_config(self.config_path)
                config.update(
                    {
                        "output_dir": str(self._output_dir),
                        "max_steps": DEBUG_STEPS,
                        "wandb_enabled": False,
                        "wandb_mode": "disabled",
                        "wandb_run_name": None,
                        "wandb_run_id": None,
                    }
                )
                trainer = OverfitTrainer(
                    config,
                    device=DEBUG_DEVICE,
                    wandb_overrides={"wandb_enabled": False, "wandb_mode": "disabled"},
                    debug_mode=True,
                    debug_tracer=tracer,
                )
                trace_event(
                    "session",
                    "session.model_optimizer_ready",
                    details={
                        "layout": trainer.model.layout.name,
                        "sequence_length": trainer.model.layout.sequence_length,
                        **trainer.model.parameter_manifest(),
                    },
                )
                for step in range(1, DEBUG_STEPS + 1):
                    tracer.set_step(step, DEBUG_STEPS)
                    trainer.train_step()

                trace_event(
                    "session",
                    "session.five_steps_complete",
                    details={
                        "global_step": trainer.global_step,
                        "metrics_path": str(trainer.metrics_path),
                        "event_log": str(self._output_dir / "events.jsonl"),
                    },
                    pause=False,
                )
                exit_status = "completed"
        except DebugRunStopped as error:
            tracer.set_status("stopped", str(error))
            exit_status = "stopped"
        except Exception as error:
            try:
                tracer.emit(
                    "error",
                    "session.training_error",
                    details={
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    phase="error",
                    pause=False,
                )
            finally:
                tracer.set_status("error", f"{type(error).__name__}: {error}")
            exit_status = "error"
        finally:
            trainer = None
            if cuda_initialized:
                try:
                    with torch.cuda.device(DEBUG_DEVICE):
                        torch.cuda.empty_cache()
                except (RuntimeError, ValueError):
                    pass
            if exit_status == "completed":
                tracer.set_status("completed", "Five optimizer steps completed")
            tracer.close()
