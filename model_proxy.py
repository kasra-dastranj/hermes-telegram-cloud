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
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


LISTEN_PORT = int(os.environ.get("MODEL_PROXY_PORT", "20129"))
ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "20128"))
MODEL_ATTEMPT_TIMEOUT = int(os.environ.get("MODEL_ATTEMPT_TIMEOUT", "20"))
MODEL_HISTORY_MAX_CHARS = int(os.environ.get("MODEL_HISTORY_MAX_CHARS", "24000"))
GROQ_HISTORY_MAX_CHARS = int(os.environ.get("GROQ_HISTORY_MAX_CHARS", "12000"))
FALLBACK_MODELS_FILE = os.environ.get(
    "FALLBACK_MODELS_FILE", "/opt/data/9router/fallback-models.json"
)
COMBO_MODEL = os.environ.get("COMBO_MODEL", "hermes-free")
GROQ_HISTORY_METADATA = {
    "reasoning_details",
    "reasoning_content",
    "reasoning",
    "thinking",
    "thinking_blocks",
}
MODEL_COOLDOWNS: dict[str, float] = {}
MODEL_COOLDOWNS_LOCK = threading.Lock()


def _compact_message(message: Any) -> Any:
    """Bound giant tool/terminal output while retaining its useful edges."""
    if not isinstance(message, dict):
        return message
    compacted = dict(message)
    content = compacted.get("content")
    limit = 4000 if compacted.get("role") == "tool" else 10000
    if isinstance(content, str) and len(content) > limit:
        edge = max(1, (limit - 120) // 2)
        compacted["content"] = (
            content[:edge]
            + "\n...[older oversized content trimmed by gateway]...\n"
            + content[-edge:]
        )
    return compacted


def bounded_messages(messages: list[Any], max_chars: int) -> list[Any]:
    """Keep system instructions and the newest coherent history within a budget."""
    compacted = [_compact_message(message) for message in messages]
    system = [
        message
        for message in compacted
        if isinstance(message, dict) and message.get("role") == "system"
    ]
    conversation = [message for message in compacted if message not in system]
    used = len(json.dumps(system, ensure_ascii=False, default=str))
    selected: list[Any] = []
    for message in reversed(conversation):
        cost = len(json.dumps(message, ensure_ascii=False, default=str))
        if selected and used + cost > max_chars:
            break
        selected.append(message)
        used += cost
    selected.reverse()
    while (
        selected
        and isinstance(selected[0], dict)
        and selected[0].get("role") == "tool"
    ):
        selected.pop(0)
    return system + selected


def model_cooldown_seconds(status: int | None) -> int:
    if status in {400, 401, 403, 404}:
        return 3600
    if status == 429:
        return 120
    if status == 413:
        return 30
    if status is None or status >= 500:
        return 300
    return 0


def model_is_cooling_down(model: str) -> bool:
    with MODEL_COOLDOWNS_LOCK:
        expires = MODEL_COOLDOWNS.get(model, 0)
        if expires <= time.monotonic():
            MODEL_COOLDOWNS.pop(model, None)
            return False
        return True


def cool_down_model(model: str, status: int | None) -> None:
    seconds = model_cooldown_seconds(status)
    if not seconds:
        return
    with MODEL_COOLDOWNS_LOCK:
        MODEL_COOLDOWNS[model] = time.monotonic() + seconds


def load_fallback_models() -> list[str]:
    """Load the exact enabled-model order written by configure_9router.js."""
    with open(FALLBACK_MODELS_FILE, encoding="utf-8") as manifest:
        models = json.load(manifest)
    if (
        not isinstance(models, list)
        or not models
        or any(not isinstance(model, str) or not model.strip() for model in models)
    ):
        raise ValueError("fallback model manifest is invalid")
    return [model.strip() for model in models]


def request_for_model(
    request_payload: dict[str, Any], model: str
) -> dict[str, Any]:
    """Build a provider-compatible attempt without mutating Hermes history.

    Hermes preserves reasoning metadata returned by some OpenAI-compatible
    providers. Groq rejects those non-standard properties when the same
    conversation later falls back to one of its models. The metadata is not
    needed to continue the visible conversation, so remove it only from the
    Groq attempt while preserving content, tool calls, and the source payload.
    """
    attempt_payload = {**request_payload, "model": model, "stream": False}
    messages = request_payload.get("messages")
    if isinstance(messages, list):
        budget = (
            GROQ_HISTORY_MAX_CHARS
            if model.startswith("groq/")
            else MODEL_HISTORY_MAX_CHARS
        )
        sanitized_messages = bounded_messages(messages, budget)
    else:
        sanitized_messages = None

    if model.startswith("groq/") and sanitized_messages is not None:
        sanitized_messages = [
            (
                {
                    key: value
                    for key, value in message.items()
                    if key not in GROQ_HISTORY_METADATA
                }
                if isinstance(message, dict)
                else message
            )
            for message in sanitized_messages
        ]
        requested_max = attempt_payload.get("max_tokens", 1024)
        try:
            attempt_payload["max_tokens"] = min(int(requested_max), 1024)
        except (TypeError, ValueError):
            attempt_payload["max_tokens"] = 1024
    elif sanitized_messages is not None:
        requested_max = attempt_payload.get("max_tokens", 2048)
        try:
            attempt_payload["max_tokens"] = min(int(requested_max), 2048)
        except (TypeError, ValueError):
            attempt_payload["max_tokens"] = 2048

    if sanitized_messages is not None:
        attempt_payload["messages"] = sanitized_messages
    return attempt_payload


def choice_has_output(choice: dict[str, Any]) -> bool:
    """Reject provider HTTP-200 responses that contain no usable assistant output."""
    message = choice.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return True
    if isinstance(message.get("tool_calls"), list) and message["tool_calls"]:
        return True
    return any(message.get(key) for key in ("function_call", "refusal"))


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
        if not choice_has_output(choice):
            raise ValueError("completed response contains no usable output")
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

    def _router_request(
        self, method: str, path: str, raw_body: bytes, headers: dict[str, str]
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        connection = http.client.HTTPConnection(
            "127.0.0.1", ROUTER_PORT, timeout=MODEL_ATTEMPT_TIMEOUT
        )
        try:
            connection.request(
                method, path, body=raw_body or None, headers=headers
            )
            response = connection.getresponse()
            return response.status, response.getheaders(), response.read()
        finally:
            connection.close()

    def _send_json_error(self, status: int, message: str) -> None:
        body = json.dumps(
            {"error": {"message": message, "type": "model_proxy_error"}}
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self, status: int, response_body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def _fallback_completion(
        self,
        request_payload: dict[str, Any],
        forwarded_headers: dict[str, str],
    ) -> None:
        try:
            models = load_fallback_models()
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json_error(503, f"Fallback manifest unavailable: {error}")
            return

        failures: list[str] = []
        for position, model in enumerate(models, start=1):
            if model_is_cooling_down(model):
                failures.append(f"{model}: cooling down")
                print(
                    f"[model-proxy] Skipping cooling-down model: {model}",
                    flush=True,
                )
                continue
            attempt_payload = request_for_model(request_payload, model)
            attempt_body = json.dumps(
                attempt_payload, ensure_ascii=False
            ).encode("utf-8")
            attempt_headers = {
                **forwarded_headers,
                "Content-Length": str(len(attempt_body)),
            }
            print(
                f"[model-proxy] Trying fallback model {position}/{len(models)}: {model}",
                flush=True,
            )
            try:
                status, _headers, response_body = self._router_request(
                    "POST", self.path, attempt_body, attempt_headers
                )
            except (ConnectionError, OSError, TimeoutError, socket.timeout,
                    http.client.HTTPException) as error:
                failures.append(f"{model}: {type(error).__name__}")
                cool_down_model(model, None)
                print(
                    f"[model-proxy] Model {model} failed: {type(error).__name__}",
                    flush=True,
                )
                continue

            if 200 <= status < 300:
                try:
                    completed = json.loads(response_body)
                    stream_body = response_to_sse(completed)
                except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                    failures.append(f"{model}: invalid response")
                    print(
                        f"[model-proxy] Model {model} returned invalid output: {error}",
                        flush=True,
                    )
                    continue
                print(f"[model-proxy] Selected fallback model: {model}", flush=True)
                with MODEL_COOLDOWNS_LOCK:
                    MODEL_COOLDOWNS.pop(model, None)
                self._send_stream(status, stream_body)
                return

            failures.append(f"{model}: HTTP {status}")
            cool_down_model(model, status)
            print(
                f"[model-proxy] Model {model} failed with HTTP {status}",
                flush=True,
            )

        summary = "; ".join(failures[-3:]) or "no model attempts completed"
        self._send_json_error(
            502, f"All {len(models)} fallback models failed ({summary})"
        )

    def _relay(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length else b""
        requested_stream = False

        is_completion = method == "POST" and self.path.split("?", 1)[0].endswith(
            "/chat/completions"
        )
        request_payload: dict[str, Any] | None = None
        if is_completion:
            try:
                request_payload = json.loads(raw_body)
            except (TypeError, ValueError):
                self._send_json_error(400, "Invalid JSON")
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

        try:
            if (
                is_completion
                and requested_stream
                and request_payload is not None
                and request_payload.get("model") == COMBO_MODEL
            ):
                self._fallback_completion(request_payload, forwarded_headers)
                return

            status, response_headers, response_body = self._router_request(
                method, self.path, raw_body, forwarded_headers
            )

            if requested_stream and 200 <= status < 300:
                try:
                    completed = json.loads(response_body)
                    response_body = response_to_sse(completed)
                except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                    self._send_json_error(502, "Invalid completed model response")
                    return
                self._send_stream(status, response_body)
                return

            self.send_response(status)
            for key, value in response_headers:
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
        except (ConnectionError, OSError, TimeoutError, socket.timeout,
                http.client.HTTPException):
            self._send_json_error(502, "9Router unavailable")

    def do_GET(self) -> None:  # noqa: N802
        self._relay("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._relay("POST")

    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        print("[model-proxy] " + (format % args), flush=True)


if __name__ == "__main__":
    print(
        f"[model-proxy] Listening on 127.0.0.1:{LISTEN_PORT}; "
        f"buffering 9Router on 127.0.0.1:{ROUTER_PORT}",
        flush=True,
    )
    ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler).serve_forever()
