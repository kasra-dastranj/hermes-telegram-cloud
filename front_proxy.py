"""Immediate public listener and reverse proxy for Hermes' Telegram webhook."""

from __future__ import annotations

import http.client
import json
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


PUBLIC_PORT = int(os.environ.get("PUBLIC_PORT", "7860"))
HERMES_PORT = int(os.environ.get("TELEGRAM_WEBHOOK_PORT", "8443"))
ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "20128"))
MODEL_PROXY_PORT = int(os.environ.get("MODEL_PROXY_PORT", "20129"))


def configured_webhook_path(webhook_url: str) -> str:
    """Return the route Hermes registers for its Telegram webhook."""
    path = urlsplit(webhook_url).path.rstrip("/")
    return path if path.startswith("/") and path else "/telegram"


WEBHOOK_PATH = configured_webhook_path(
    os.environ.get("TELEGRAM_WEBHOOK_URL", "")
)


def local_public_path(path: str, webhook_path: str = WEBHOOK_PATH) -> str:
    """Map an externally namespaced health/root path to the local endpoint."""
    webhook_prefix = webhook_path.rsplit("/", 1)[0]
    if webhook_prefix and (
        path == webhook_prefix or path.startswith(f"{webhook_prefix}/")
    ):
        return path[len(webhook_prefix) :] or "/"
    return path


def port_ready(port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def component_health() -> dict[str, bool]:
    return {
        "hermes": port_ready(HERMES_PORT),
        "router": port_ready(ROUTER_PORT),
        "model_proxy": port_ready(MODEL_PROXY_PORT),
    }


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
        path = local_public_path(self.path.split("?", 1)[0])
        if path == "/":
            self._send_json(
                200,
                {
                    "ok": True,
                    "service": "hermes-telegram",
                    "message": "Public gateway is awake",
                },
            )
            return
        if path == "/health":
            components = component_health()
            healthy = all(components.values())
            self._send_json(
                200 if healthy else 503,
                {
                    "ok": healthy,
                    "service": "hermes-telegram",
                    **{
                        name: "ready" if ready else "unavailable"
                        for name, ready in components.items()
                    },
                },
            )
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_HEAD(self) -> None:  # noqa: N802
        path = local_public_path(self.path.split("?", 1)[0])
        if path in ("/", "/health"):
            healthy = path == "/" or all(component_health().values())
            self.send_response(200 if healthy else 503)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        request_path = self.path.split("?", 1)[0]
        if request_path not in {"/telegram", WEBHOOK_PATH}:
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
            connection = http.client.HTTPConnection("127.0.0.1", HERMES_PORT, timeout=30)
            connection.request("POST", WEBHOOK_PATH, body=body, headers=forwarded_headers)
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
        except (ConnectionError, OSError, TimeoutError, http.client.HTTPException):
            # Telegram retries non-2xx webhook deliveries. This is expected
            # briefly while a free Render instance is waking up.
            self._send_json(503, {"ok": False, "error": "hermes_starting"})

    def log_message(self, format: str, *args: object) -> None:
        print("[proxy] " + (format % args), flush=True)


if __name__ == "__main__":
    print(
        f"[proxy] Listening on 0.0.0.0:{PUBLIC_PORT}; "
        f"forwarding {WEBHOOK_PATH} to 127.0.0.1:{HERMES_PORT}",
        flush=True,
    )
    ThreadingHTTPServer(("0.0.0.0", PUBLIC_PORT), Handler).serve_forever()
