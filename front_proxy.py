"""Immediate public listener and reverse proxy for Hermes' Telegram webhook."""

from __future__ import annotations

import http.client
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PUBLIC_PORT = int(os.environ.get("PUBLIC_PORT", "7860"))
HERMES_PORT = int(os.environ.get("TELEGRAM_WEBHOOK_PORT", "8443"))


class Handler(BaseHTTPRequestHandler):
    server_version = "HermesRenderProxy/1.0"

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/health"):
            self._send_json(
                200,
                {
                    "ok": True,
                    "service": "hermes-telegram",
                    "message": "Public gateway is awake",
                },
            )
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_HEAD(self) -> None:  # noqa: N802
        if self.path in ("/", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/telegram":
            self._send_json(404, {"ok": False, "error": "not_found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        forwarded_headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "connection", "content-length"}
        }
        forwarded_headers["Content-Length"] = str(len(body))

        try:
            connection = http.client.HTTPConnection(
                "127.0.0.1", HERMES_PORT, timeout=15
            )
            connection.request("POST", "/telegram", body=body, headers=forwarded_headers)
            response = connection.getresponse()
            response_body = response.read()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in {
                    "connection",
                    "content-length",
                    "transfer-encoding",
                }:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
            connection.close()
        except (ConnectionError, OSError, TimeoutError):
            # Telegram retries non-2xx webhook deliveries. This is expected
            # briefly while a free Render instance is waking up.
            self._send_json(503, {"ok": False, "error": "hermes_starting"})

    def log_message(self, format: str, *args: object) -> None:
        print("[proxy] " + (format % args), flush=True)


if __name__ == "__main__":
    print(
        f"[proxy] Listening on 0.0.0.0:{PUBLIC_PORT}; "
        f"forwarding /telegram to 127.0.0.1:{HERMES_PORT}",
        flush=True,
    )
    ThreadingHTTPServer(("0.0.0.0", PUBLIC_PORT), Handler).serve_forever()
