from __future__ import annotations

import unittest

from Full_Duplex_Fix.debug.server import create_app


class _FakeManager:
    def __init__(self) -> None:
        self.status = "not_started"

    def state(self):
        return {
            "status": self.status,
            "step": 0,
            "total_steps": 5,
            "event_count": 0,
            "device": "cuda:2",
        }

    def events_after(self, event_id, limit=200):
        del event_id, limit
        return []

    def start(self):
        self.status = "paused"
        return self.state()

    def control(self, action, delay_ms=None):
        del delay_ms
        self.status = "running" if action in {"next", "auto"} else self.status
        return self.state()

    def reset(self):
        self.status = "not_started"
        return self.state()


class DebugServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = _FakeManager()
        self.client = create_app(self.manager).test_client()

    def test_page_and_static_assets_are_served(self) -> None:
        for path in ("/", "/static/styles.css", "/static/app.js"):
            response = self.client.get(path)
            try:
                self.assertEqual(response.status_code, 200)
            finally:
                response.close()

    def test_start_poll_and_control_protocol(self) -> None:
        self.assertEqual(self.client.get("/api/state").json["status"], "not_started")
        self.assertEqual(self.client.post("/api/start").json["status"], "paused")
        response = self.client.get("/api/events?after=0&limit=20").json
        self.assertEqual(response["events"], [])
        self.assertEqual(
            self.client.post("/api/control", json={"action": "next"}).json["status"],
            "running",
        )


if __name__ == "__main__":
    unittest.main()
