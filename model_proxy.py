"""Buffer 9Router responses, then expose them as a reliable OpenAI SSE stream.

9Router can only try the next fallback model before response headers have been
sent. Hermes, meanwhile, expects a streaming response in gateway mode. This
loopback-only adapter lets 9Router finish a non-streaming request (including
fallbacks) and then converts the completed OpenAI response into valid SSE.
"""

from __future__ import annotations

import http.client
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


LISTEN_PORT = int(os.environ.get("MODEL_PROXY_PORT", "20129"))
ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "20128"))


def response_to_sse(payload: dict[str, Any]) -> bytes:
    """Convert one completed Chat Completions response to OpenAI SSE chunks."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("completed response has no choices")

    base = {
        "id": payload.get("id", "chatcmpl-buffered"),
        "object": "chat.completion.chunk",
        "created": payload.get("created", 0),
        "model": payload.get("model", ""),
    }
    content_choices = []
    finish_choices = []

    for position, choice in enumerate(choices):
        if not isinstance(choice, dict) or not isinstance(
            choice.get("message"), dict
        ):
            raise ValueError("completed response contains an invalid choice")
        index = choice.get("index", position)
        message = dict(choice.get("message") or {})
        delta: dict[str, Any] = {}
        for key in (
            "role",
            "content",
            "reasoning_content",
            "reasoning",
            "function_call",
            "refusal",
        ):
            if key in message:
                delta[key] = message[key]

        if isinstance(message.get("tool_calls"), list):
            delta["tool_calls"] = [
                {**tool_call, "index": tool_index}
                for tool_index, tool_call in enumerate(message["tool_calls"])
            ]

        content_choices.append(
            {"index": index, "delta": delta, "finish_reason": None}
        )
        finish_choices.append(
            {
                "index": index,
                "delta": {},
                "finish_reason": choice.get("finish_reason") or "stop",
            }
        )

    chunks = [
        json.dumps({**base, "choices": content_choices}, ensure_ascii=False),
        json.dumps({**base, "choices": finish_choices}, ensure_ascii=False),
    ]
    if payload.get("usage") is not None:
        chunks.append(
            json.dumps(
                {**base, "choices": [], "usage": payload["usage"]},
                ensure_ascii=False,
            )
        )
    return ("".join(f"data: {chunk}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode(
        "utf-8"
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "HermesModelBuffer/1.0"

    def _relay(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length else b""
        requested_stream = False

        if method == "POST" and self.path.split("?", 1)[0].endswith(
            "/chat/completions"
        ):
            try:
                request_payload = json.loads(raw_body)
            except (TypeError, ValueError):
                self.send_error(400, "Invalid JSON")
                return
            requested_stream = bool(request_payload.get("stream"))
            request_payload["stream"] = False
            request_payload.pop("stream_options", None)
            raw_body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")

        forwarded_headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower()
            not in {"host", "connection", "content-length", "accept-encoding"}
        }
        if raw_body:
            forwarded_headers["Content-Length"] = str(len(raw_body))

        connection = http.client.HTTPConnection(
            "127.0.0.1", ROUTER_PORT, timeout=300
        )
        try:
            connection.request(
                method, self.path, body=raw_body or None, headers=forwarded_headers
            )
            response = connection.getresponse()
            response_body = response.read()

            if requested_stream and 200 <= response.status < 300:
                try:
                    completed = json.loads(response_body)
                    response_body = response_to_sse(completed)
                except (TypeError, ValueError, KeyError):
                    self.send_error(502, "Invalid completed model response")
                    return
                self.send_response(response.status)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
                return

            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in {
                    "connection",
                    "content-length",
                    "transfer-encoding",
                    "content-encoding",
                }:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        except (ConnectionError, OSError, TimeoutError, http.client.HTTPException):
            self.send_error(502, "9Router unavailable")
        finally:
            connection.close()

    def do_GET(self) -> None:  # noqa: N802
        self._relay("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._relay("POST")

    def log_message(self, format: str, *args: object) -> None:
        print("[model-proxy] " + (format % args), flush=True)


if __name__ == "__main__":
    print(
        f"[model-proxy] Listening on 127.0.0.1:{LISTEN_PORT}; "
        f"buffering 9Router on 127.0.0.1:{ROUTER_PORT}",
        flush=True,
    )
    ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler).serve_forever()
