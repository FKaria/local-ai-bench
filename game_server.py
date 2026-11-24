"""In-memory localhost HTTP game server + thin workspace CLI client (`play.py`).

Keeping the GameEnv instance strictly inside run_bench.py's process memory guarantees
the model cannot inspect hidden game state (secret codes, mine layouts, etc.) on disk,
even with full bash/read/python access inside its isolated temp workspace.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from game_envs import GameEnv


class LocalGameServer:
    """Hosts a single GameEnv instance over 127.0.0.1 on an ephemeral port."""

    def __init__(self, env: GameEnv) -> None:
        self.env = env
        self.lock = threading.Lock()
        self.initial_obs = env.reset()
        self.last_obs = self.initial_obs
        self.action_log: list[dict[str, Any]] = []
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port: int = 0

    def start(self) -> int:
        parent = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return  # silence HTTP access logs

            def _send_json(self, code: int, payload: dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/status":
                    with parent.lock:
                        self._send_json(200, {"observation": parent.last_obs, "status": parent.env.status()})
                else:
                    self._send_json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/action":
                    self._send_json(404, {"error": "not found"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8", errors="replace")
                try:
                    data = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    self._send_json(400, {"error": "invalid json"})
                    return
                action = str(data.get("action", "")).strip()
                with parent.lock:
                    obs, done, st = parent.env.step(action)
                    parent.last_obs = obs
                    parent.action_log.append({
                        "turn": st["turns"],
                        "action": action,
                        "done": done,
                        "won": st["won"],
                        "progress": st["progress"],
                    })
                    self._send_json(200, {"observation": obs, "done": done, "status": st})

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None


def render_play_client(port: int) -> str:
    """Generate the thin `play.py` CLI placed inside the model's isolated workspace."""
    return f'''"""Thin CLI client for playing the local benchmark game server.

Usage:
    python play.py status
    python play.py <action...>
"""

import json
import sys
import urllib.request

BASE_URL = "http://127.0.0.1:{port}"


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0].lower() in ("status", "board", "show"):
        with urllib.request.urlopen(f"{{BASE_URL}}/status", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        print(data["observation"])
        return 0

    if args[0].lower() in ("move", "act", "action") and len(args) > 1:
        action = " ".join(args[1:])
    else:
        action = " ".join(args)

    payload = json.dumps({{"action": action}}).encode("utf-8")
    req = urllib.request.Request(
        f"{{BASE_URL}}/action",
        data=payload,
        headers={{"Content-Type": "application/json"}},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    print(data["observation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
