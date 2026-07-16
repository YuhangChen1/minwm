from __future__ import annotations

import json
import math
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch


class DebugRunStopped(RuntimeError):
    pass


_ACTIVE_TRACER: ContextVar[DebugTracer | None] = ContextVar(
    "full_duplex_debug_tracer", default=None
)


def active_debug_tracer() -> "DebugTracer | None":
    return _ACTIVE_TRACER.get()


@contextmanager
def debug_scope(tracer: "DebugTracer | None") -> Iterator[None]:
    token = _ACTIVE_TRACER.set(tracer)
    try:
        yield
    finally:
        _ACTIVE_TRACER.reset(token)


def _json_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def first_values(tensor: torch.Tensor | None, count: int = 2) -> list[float | None]:
    if tensor is None or tensor.numel() == 0:
        return []
    values = tensor.detach().reshape(-1)[:count].float().cpu().tolist()
    return [_json_float(value) for value in values]


def tensor_summary(tensor: torch.Tensor, sample_size: int = 2048) -> dict[str, Any]:
    detached = tensor.detach()
    result: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": detached.numel(),
        "requires_grad": bool(tensor.requires_grad),
    }
    if detached.numel() == 0:
        result.update({"first_values": [], "sample_size": 0, "finite": True})
        return result

    flat = detached.reshape(-1)
    stride = max(1, flat.numel() // sample_size)
    first_count = min(2, flat.numel())
    sampled_gpu = flat[::stride][:sample_size].float()
    packed = torch.cat((flat[:first_count].float(), sampled_gpu)).cpu()
    first = packed[:first_count]
    sample = packed[first_count:]
    finite = torch.isfinite(sample)
    finite_sample = sample[finite]
    result["first_values"] = [_json_float(value) for value in first.tolist()]
    result["sample_size"] = int(sample.numel())
    result["sample_stride"] = int(stride)
    result["finite"] = bool(finite.all())
    if finite_sample.numel():
        result.update(
            {
                "min": _json_float(finite_sample.min()),
                "max": _json_float(finite_sample.max()),
                "mean": _json_float(finite_sample.mean()),
                "std": _json_float(finite_sample.std(unbiased=False)),
                "l2": _json_float(torch.linalg.vector_norm(finite_sample)),
            }
        )
    return result


def summarize_tensors(tensors: Mapping[str, torch.Tensor] | None) -> dict[str, Any]:
    if not tensors:
        return {}
    return {name: tensor_summary(tensor) for name, tensor in tensors.items()}


@contextmanager
def debug_timer() -> Iterator[dict[str, float]]:
    tracer = active_debug_tracer()
    timing: dict[str, float] = {}
    if tracer is not None and tracer.synchronize_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        yield timing
    finally:
        if tracer is not None and tracer.synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        timing["duration_ms"] = (time.perf_counter() - started) * 1000.0


class DebugTracer:
    def __init__(
        self,
        *,
        log_path: str | Path | None = None,
        mode: str = "step",
        synchronize_cuda: bool = True,
        auto_delay_ms: int = 0,
    ) -> None:
        if mode not in {"step", "auto"}:
            raise ValueError(f"Unsupported debug mode: {mode}")
        self.mode = mode
        self.synchronize_cuda = bool(synchronize_cuda)
        self.auto_delay_ms = max(0, int(auto_delay_ms))
        self.log_path = Path(log_path).resolve() if log_path is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")

        self._condition = threading.Condition()
        self._events: list[dict[str, Any]] = []
        self._next_id = 1
        self._permits = 0
        self._stop_requested = False
        self._status = "idle"
        self._status_message = ""
        self._current_event_id: int | None = None
        self._step = 0
        self._total_steps = 5
        self._runtime_phase = "forward"
        self._started_at = time.time()

    @property
    def step(self) -> int:
        with self._condition:
            return self._step

    def set_step(self, step: int, total_steps: int = 5) -> None:
        with self._condition:
            self._step = int(step)
            self._total_steps = int(total_steps)

    @property
    def runtime_phase(self) -> str:
        with self._condition:
            return self._runtime_phase

    def set_runtime_phase(self, phase: str) -> None:
        with self._condition:
            self._runtime_phase = phase

    def set_status(self, status: str, message: str = "") -> None:
        with self._condition:
            self._status = status
            self._status_message = message
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "status": self._status,
                "message": self._status_message,
                "mode": self.mode,
                "step": self._step,
                "total_steps": self._total_steps,
                "runtime_phase": self._runtime_phase,
                "event_count": len(self._events),
                "current_event_id": self._current_event_id,
                "stop_requested": self._stop_requested,
                "started_at": self._started_at,
            }

    def events_after(self, event_id: int, limit: int = 200) -> list[dict[str, Any]]:
        with self._condition:
            return [event for event in self._events if event["id"] > event_id][:limit]

    def continue_one(self) -> None:
        with self._condition:
            self.mode = "step"
            self._permits += 1
            self._status = "running"
            self._condition.notify_all()

    def run_auto(self, delay_ms: int | None = None) -> None:
        with self._condition:
            self.mode = "auto"
            if delay_ms is not None:
                self.auto_delay_ms = max(0, int(delay_ms))
            self._status = "running"
            self._condition.notify_all()

    def pause_at_next(self) -> None:
        with self._condition:
            self.mode = "step"
            self._permits = 0
            self._condition.notify_all()

    def request_stop(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._status = "stopping"
            self._status_message = "Stopping at the next debug boundary"
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def emit(
        self,
        category: str,
        name: str,
        *,
        tensors: Mapping[str, torch.Tensor] | None = None,
        details: Mapping[str, Any] | None = None,
        phase: str = "forward",
        pause: bool = True,
    ) -> int:
        event = {
            "id": 0,
            "timestamp": time.time(),
            "elapsed_seconds": time.time() - self._started_at,
            "step": self.step,
            "phase": phase,
            "category": category,
            "name": name,
            "details": dict(details or {}),
            "tensors": summarize_tensors(tensors),
        }
        with self._condition:
            if self._stop_requested:
                raise DebugRunStopped("Debug training stopped by the user")
            event["id"] = self._next_id
            self._next_id += 1
            self._events.append(event)
            self._current_event_id = event["id"]
            if self.log_path is not None:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
                    handle.write("\n")
            self._condition.notify_all()

            if pause and self.mode == "step":
                self._status = "paused"
                self._status_message = name
                while self.mode == "step" and self._permits <= 0 and not self._stop_requested:
                    self._condition.wait()
                if self._stop_requested:
                    raise DebugRunStopped("Debug training stopped by the user")
                if self.mode == "step":
                    self._permits -= 1
                self._status = "running"
                self._status_message = ""

        if pause and self.mode == "auto" and self.auto_delay_ms:
            time.sleep(self.auto_delay_ms / 1000.0)
        return int(event["id"])


def trace_event(
    category: str,
    name: str,
    *,
    tensors: Mapping[str, torch.Tensor] | None = None,
    details: Mapping[str, Any] | None = None,
    phase: str = "forward",
    pause: bool = True,
) -> int | None:
    tracer = active_debug_tracer()
    if tracer is None:
        return None
    if phase == "forward" and tracer.runtime_phase != "forward":
        return None
    return tracer.emit(
        category,
        name,
        tensors=tensors,
        details=details,
        phase=phase,
        pause=pause,
    )


def trace_tensor(
    category: str,
    name: str,
    tensor: torch.Tensor,
    *,
    details: Mapping[str, Any] | None = None,
    backward: bool = False,
    pause: bool = True,
) -> torch.Tensor:
    tracer = active_debug_tracer()
    if tracer is None:
        return tensor
    if tracer.runtime_phase != "forward":
        return tensor
    event_id = tracer.emit(
        category,
        name,
        tensors={"output": tensor},
        details=details,
        phase="forward",
        pause=pause,
    )
    if backward and tensor.requires_grad:
        forward_step = tracer.step

        def report_gradient(gradient: torch.Tensor) -> torch.Tensor:
            tracer.emit(
                category,
                f"{name}.gradient",
                tensors={"gradient": gradient},
                details={"forward_event_id": event_id, "forward_step": forward_step},
                phase="backward",
                pause=True,
            )
            return gradient

        tensor.register_hook(report_gradient)
    return tensor


def trace_gradient(
    category: str,
    name: str,
    tensor: torch.Tensor,
    *,
    details: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    tracer = active_debug_tracer()
    if tracer is None or tracer.runtime_phase != "forward" or not tensor.requires_grad:
        return tensor
    forward_step = tracer.step

    def report_gradient(gradient: torch.Tensor) -> torch.Tensor:
        tracer.emit(
            category,
            f"{name}.gradient",
            tensors={"gradient": gradient},
            details={"forward_step": forward_step, **dict(details or {})},
            phase="backward",
            pause=True,
        )
        return gradient

    tensor.register_hook(report_gradient)
    return tensor


def capture_parameter_values(
    parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict[str, dict[str, Any]]:
    captured: dict[str, dict[str, Any]] = {}
    for name, parameter in parameters:
        if not parameter.requires_grad or parameter.grad is None:
            continue
        captured[name] = {
            "shape": list(parameter.shape),
            "numel": parameter.numel(),
            "dtype": str(parameter.dtype),
            "before": first_values(parameter),
            "gradient": first_values(parameter.grad),
        }
    return captured


def parameter_updates(
    before: Mapping[str, Mapping[str, Any]],
    parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> list[dict[str, Any]]:
    current = dict(parameters)
    updates = []
    for name, entry in before.items():
        parameter = current[name]
        after = first_values(parameter)
        previous = list(entry["before"])
        delta = [
            None if left is None or right is None else right - left
            for left, right in zip(previous, after)
        ]
        updates.append(
            {
                "name": name,
                "shape": list(entry["shape"]),
                "numel": int(entry["numel"]),
                "dtype": entry["dtype"],
                "before": previous,
                "after": after,
                "delta": delta,
                "gradient": list(entry["gradient"]),
                "first_values_changed": any(value not in {0.0, None} for value in delta),
            }
        )
    return updates
