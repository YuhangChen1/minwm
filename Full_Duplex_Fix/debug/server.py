from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .runner import DebugRunManager


STATIC_ROOT = Path(__file__).resolve().parent / "static"


def create_app(manager: DebugRunManager | None = None) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_ROOT), static_url_path="/static")
    app.json.sort_keys = False
    run_manager = manager or DebugRunManager()

    @app.after_request
    def disable_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return send_from_directory(STATIC_ROOT, "index.html")

    @app.get("/api/state")
    def state():
        return jsonify(run_manager.state())

    @app.get("/api/events")
    def events():
        after = max(0, request.args.get("after", default=0, type=int))
        limit = min(500, max(1, request.args.get("limit", default=200, type=int)))
        return jsonify(
            {
                "events": run_manager.events_after(after, limit),
                "state": run_manager.state(),
            }
        )

    @app.post("/api/start")
    def start():
        return jsonify(run_manager.start())

    @app.post("/api/control")
    def control():
        payload = request.get_json(silent=True) or {}
        try:
            result = run_manager.control(
                str(payload.get("action", "")),
                delay_ms=payload.get("delay_ms"),
            )
        except (RuntimeError, ValueError) as error:
            return jsonify({"error": str(error), "state": run_manager.state()}), 409
        return jsonify(result)

    @app.post("/api/reset")
    def reset():
        try:
            result = run_manager.reset()
        except RuntimeError as error:
            return jsonify({"error": str(error), "state": run_manager.state()}), 409
        return jsonify(result)

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "state": run_manager.state()})

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", default="Full_Duplex_Fix/configs/overfit.yaml")
    args = parser.parse_args()
    manager = DebugRunManager(config_path=args.config)
    create_app(manager).run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
