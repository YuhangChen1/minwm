from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import torch

from Full_Duplex_Fix.debug.tracer import (
    DebugTracer,
    capture_parameter_values,
    debug_scope,
    parameter_updates,
    trace_event,
)


class DebugTracerTest(unittest.TestCase):
    def test_step_mode_blocks_until_next(self) -> None:
        tracer = DebugTracer(mode="step", synchronize_cuda=False)
        completed = threading.Event()

        def publish() -> None:
            tracer.emit("test", "first")
            completed.set()

        thread = threading.Thread(target=publish)
        thread.start()
        deadline = time.time() + 2
        while tracer.snapshot()["event_count"] == 0 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(tracer.snapshot()["status"], "paused")
        self.assertFalse(completed.is_set())
        tracer.continue_one()
        thread.join(timeout=2)
        self.assertTrue(completed.is_set())

    def test_runtime_phase_suppresses_checkpoint_recompute_events(self) -> None:
        tracer = DebugTracer(mode="auto", synchronize_cuda=False)
        with debug_scope(tracer):
            trace_event("test", "forward")
            tracer.set_runtime_phase("backward")
            trace_event("test", "recomputed_forward")
        events = tracer.events_after(0)
        self.assertEqual([event["name"] for event in events], ["forward"])

    def test_parameter_update_records_first_two_values(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        parameter.grad = torch.tensor([0.5, -0.25, 0.0])
        before = capture_parameter_values([("weight", parameter)])
        with torch.no_grad():
            parameter.add_(torch.tensor([-0.1, 0.2, 0.0]))
        row = parameter_updates(before, [("weight", parameter)])[0]
        self.assertEqual(row["before"], [1.0, 2.0])
        self.assertAlmostEqual(row["after"][0], 0.9, places=6)
        self.assertAlmostEqual(row["delta"][1], 0.2, places=6)
        self.assertEqual(row["gradient"], [0.5, -0.25])

    def test_event_log_is_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            tracer = DebugTracer(log_path=path, mode="auto", synchronize_cuda=False)
            tracer.emit("test", "tensor", tensors={"value": torch.arange(4)})
            payload = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["name"], "tensor")
            self.assertEqual(payload["tensors"]["value"]["shape"], [4])


if __name__ == "__main__":
    unittest.main()
