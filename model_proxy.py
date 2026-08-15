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
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


LISTEN_PORT = int(os.environ.get("MODEL_PROXY_PORT", "20129"))
ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "20128"))
MODEL_ATTEMPT_TIMEOUT = int(os.environ.get("MODEL_ATTEMPT_TIMEOUT", "20"))
GOROUTER_ATTEMPT_TIMEOUT = int(
    os.environ.get("GOROUTER_ATTEMPT_TIMEOUT", "90")
)
OPENROUTER_ATTEMPT_TIMEOUT = int(
    os.environ.get("OPENROUTER_ATTEMPT_TIMEOUT", "45")
)
RATE_LIMIT_RETRIES = int(os.environ.get("RATE_LIMIT_RETRIES", "2"))
RATE_LIMIT_MAX_WAIT = float(os.environ.get("RATE_LIMIT_MAX_WAIT", "65"))
MODEL_HISTORY_MAX_CHARS = int(
    os.environ.get("MODEL_HISTORY_MAX_CHARS", "180000")
)
GOROUTER_HISTORY_MAX_CHARS = int(
    os.environ.get("GOROUTER_HISTORY_MAX_CHARS", "360000")
)
OPENROUTER_HISTORY_MAX_CHARS = int(
    os.environ.get("OPENROUTER_HISTORY_MAX_CHARS", "120000")
)
GROQ_HISTORY_MAX_CHARS = int(os.environ.get("GROQ_HISTORY_MAX_CHARS", "9000"))
FALLBACK_MODELS_FILE = os.environ.get(
    "FALLBACK_MODELS_FILE", "/opt/data/9router/fallback-models.json"
)
COMBO_MODEL = os.environ.get("COMBO_MODEL", "").strip() or "hermes-free"
GROQ_HISTORY_METADATA = {
    "reasoning_details",
    "reasoning_content",
    "reasoning",
    "thinking",
    "thinking_blocks",
}
MODEL_COOLDOWNS: dict[str, float] = {}
MODEL_COOLDOWNS_LOCK = threading.Lock()
DIRECT_PROVIDERS = {
    "gorouter": (
        "gorouter.app",
        "/v1/chat/completions",
        "GOROUTER_API_KEY",
    ),
    "groq": ("api.groq.com", "/openai/v1/chat/completions", "GROQ_API_KEY"),
    "openrouter": (
        "openrouter.ai",
        "/api/v1/chat/completions",
        "OPENROUTER_API_KEY",
    ),
}


def _compact_message(message: Any, tool_content_limit: int = 4000) -> Any:
    """Bound giant tool/terminal output while retaining its useful edges."""
    if not isinstance(message, dict):
        return message
    compacted = dict(message)
    content = compacted.get("content")
    limit = tool_content_limit if compacted.get("role") == "tool" else 10000
    if isinstance(content, str) and len(content) > limit:
        edge = max(1, (limit - 120) // 2)
        compacted["content"] = (
            content[:edge]
            + "\n...[older oversized content trimmed by gateway]...\n"
            + content[-edge:]
        )
    return compacted


def _message_cost(messages: list[Any]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, default=str))


def _conversation_turns(messages: list[Any]) -> list[list[Any]]:
    """Split history into user-led turns so trimming never starts mid-turn."""
    turns: list[list[Any]] = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            turns.append([message])
        elif turns:
            turns[-1].append(message)
    return turns


def _fit_current_turn(current_turn: list[Any], max_chars: int) -> list[Any]:
    """Retain a coherent recent suffix of an oversized active tool turn."""
    if not current_turn or _message_cost(current_turn) <= max_chars:
        return current_turn

    user = current_turn[0]
    user_cost = _message_cost([user])
    if user_cost >= max_chars and isinstance(user, dict):
        user = dict(user)
        content = user.get("content")
        if isinstance(content, str):
            keep = max(256, max_chars - 512)
            user["content"] = content[-keep:]

    # Every assistant message starts a protocol segment and owns any tool
    # results that follow it. Selecting complete recent segments avoids
    # dangling tool messages that OpenAI-compatible providers reject.
    segments: list[list[Any]] = []
    for message in current_turn[1:]:
        if isinstance(message, dict) and message.get("role") == "assistant":
            segments.append([message])
        elif segments:
            segments[-1].append(message)

    selected: list[list[Any]] = []
    used = _message_cost([user])
    for segment in reversed(segments):
        cost = _message_cost(segment)
        if used + cost > max_chars:
            continue
        selected.append(segment)
        used += cost
    selected.reverse()
    return [user, *(message for segment in selected for message in segment)]


def bounded_messages(
    messages: list[Any], max_chars: int, tool_content_limit: int = 4000
) -> list[Any]:
    """Keep system instructions and the newest coherent user turn.

    A tool continuation is only meaningful when the provider receives the user
    request, the assistant tool call, and the matching tool result together.
    The budget applies to conversation history, not the always-retained system
    prompt. Count complete user-led turns so a short follow-up such as "do it"
    keeps the immediately preceding request instead of becoming context-free.
    """
    compacted = [
        _compact_message(message, tool_content_limit=tool_content_limit)
        for message in messages
    ]
    system = [
        message
        for message in compacted
        if isinstance(message, dict) and message.get("role") == "system"
    ]
    conversation = [message for message in compacted if message not in system]
    latest_user_index = next(
        (
            index
            for index in range(len(conversation) - 1, -1, -1)
            if isinstance(conversation[index], dict)
            and conversation[index].get("role") == "user"
        ),
        None,
    )
    if latest_user_index is None:
        current_turn: list[Any] = []
        older = conversation
    else:
        current_turn = conversation[latest_user_index:]
        older = conversation[:latest_user_index]

    current_turn = _fit_current_turn(current_turn, max_chars)
    used = _message_cost(current_turn)
    selected_turns: list[list[Any]] = []
    for turn in reversed(_conversation_turns(older)):
        cost = _message_cost(turn)
        if used + cost > max_chars:
            break
        selected_turns.append(turn)
        used += cost
    selected_turns.reverse()
    selected = [message for turn in selected_turns for message in turn]
    return system + selected + current_turn


def _compact_schema(value: Any, depth: int = 0) -> Any:
    """Remove documentation-only schema bulk while keeping tool validation intact."""
    if isinstance(value, list):
        return [_compact_schema(item, depth + 1) for item in value]
    if not isinstance(value, dict):
        return value

    compacted: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"title", "examples", "example", "$comment", "default"}:
            continue
        if key == "description":
            if depth <= 2 and isinstance(item, str):
                compacted[key] = item[:240]
            continue
        compacted[key] = _compact_schema(item, depth + 1)
    return compacted


def compact_tools(tools: Any) -> Any:
    """Keep function names and JSON schemas, but trim verbose prose for Groq."""
    if not isinstance(tools, list):
        return tools
    return [_compact_schema(tool) for tool in tools]


def model_cooldown_seconds(status: int | None) -> int:
    if status == 400:
        return 30
    if status in {401, 403, 404}:
        return 3600
    if status == 429:
        return 15
    if status == 413:
        return 30
    if status is None or status >= 500:
        return 300
    return 0


def rate_limit_retry_seconds(
    response_headers: list[tuple[str, str]], response_body: bytes
) -> float | None:
    """Read a short provider-directed retry delay from headers or JSON text."""
    for key, value in response_headers:
        if key.lower() != "retry-after":
            continue
        try:
            delay = float(value)
        except (TypeError, ValueError):
            break
        return max(0.0, delay)

    text = response_body.decode("utf-8", errors="ignore")
    match = re.search(
        r"(?:try again in|retry after)\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s|sec|seconds?|m|min|minutes?)?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    delay = float(match.group(1))
    unit = (match.group(2) or "s").lower()
    if unit == "ms":
        delay /= 1000
    elif unit.startswith("m"):
        delay *= 60
    return max(0.0, delay)


def tool_call_failed(status: int, response_body: bytes) -> bool:
    if status != 400:
        return False
    text = response_body.decode("utf-8", errors="ignore").lower()
    return "tool_use_failed" in text or "tool call" in text


def add_repair_instruction(
    payload: dict[str, Any], instruction: str
) -> dict[str, Any]:
    repaired = dict(payload)
    messages = [dict(message) for message in payload.get("messages", [])]
    repair = {"role": "system", "content": instruction}
    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    messages.insert(insert_at, repair)
    repaired["messages"] = messages
    return repaired


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


def history_budget_for_model(model: str) -> int:
    """Keep rich history on Claude while protecting small free fallbacks."""
    if model.startswith("gorouter/"):
        return GOROUTER_HISTORY_MAX_CHARS
    if model.startswith("groq/"):
        return GROQ_HISTORY_MAX_CHARS
    if model.startswith("openrouter/"):
        return OPENROUTER_HISTORY_MAX_CHARS
    return MODEL_HISTORY_MAX_CHARS


def attempt_timeout_for_model(model: str) -> int:
    """Give the primary Claude route time to complete tool-heavy turns."""
    if model.startswith("gorouter/"):
        return GOROUTER_ATTEMPT_TIMEOUT
    if model.startswith("openrouter/"):
        return OPENROUTER_ATTEMPT_TIMEOUT
    return MODEL_ATTEMPT_TIMEOUT


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
        budget = history_budget_for_model(model)
        # Groq's free tier has a tight TPM ceiling. Browser and terminal
        # results can otherwise dominate the prompt even after old turns are
        # removed, so keep compact evidence snippets while preserving each
        # assistant-tool protocol segment.
        tool_content_limit = 1200 if model.startswith("groq/") else 4000
        sanitized_messages = bounded_messages(
            messages, budget, tool_content_limit=tool_content_limit
        )
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
        # Groq's free-tier TPM limit counts the prompt, tool schemas, and the
        # requested completion together. Hermes tool descriptions are large,
        # so retain their executable schemas but remove documentation-only
        # prose and reserve a modest completion budget for each agent turn.
        if "tools" in attempt_payload:
            attempt_payload["tools"] = compact_tools(attempt_payload["tools"])
        requested_max = attempt_payload.get("max_tokens", 1024)
        try:
            attempt_payload["max_tokens"] = min(int(requested_max), 1024)
        except (TypeError, ValueError):
            attempt_payload["max_tokens"] = 1024
        if "gpt-oss" in model and not attempt_payload.get("reasoning_effort"):
            attempt_payload["reasoning_effort"] = "medium"
    elif sanitized_messages is not None:
        requested_max = attempt_payload.get("max_tokens", 2048)
        try:
            attempt_payload["max_tokens"] = min(int(requested_max), 2048)
        except (TypeError, ValueError):
            attempt_payload["max_tokens"] = 2048

    # GoRouter exposes a separate "-thinking" model ID. Do not forward
    # Hermes' generic reasoning hint to the normal Claude route, otherwise a
    # short everyday turn can unexpectedly consume thousands of hidden tokens.
    if model.startswith("gorouter/"):
        attempt_payload.pop("reasoning_effort", None)

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


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(
            str(part.get("text", ""))
            for part in value
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def response_quality_error(
    payload: dict[str, Any], request_payload: dict[str, Any]
) -> str | None:
    """Detect obviously corrupted mixed-script prose before accepting fallback."""
    messages = request_payload.get("messages")
    if not isinstance(messages, list):
        return None
    latest_user = next(
        (
            _text_content(message.get("content"))
            for message in reversed(messages)
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        "",
    )
    # Apply this guard only to Persian/Arabic-script conversations. Latin text
    # and source URLs remain valid in a Persian answer.
    if len(re.findall(r"[\u0600-\u06ff]", latest_user)) < 2:
        return None

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return None
    response_text = " ".join(
        _text_content(choice.get("message", {}).get("content"))
        for choice in choices
        if isinstance(choice, dict) and isinstance(choice.get("message"), dict)
    )
    if not response_text:
        return None

    # Greek, Hebrew, Cyrillic, CJK and Hangul appearing in quantity inside a
    # Persian reply is a reliable signature of the mojibake/random-language
    # failures observed from aggregator fallbacks.
    suspicious = re.findall(
        r"[\u0370-\u052f\u0590-\u05ff\u3040-\u30ff\u3400-\u4dbf"
        r"\u4e00-\u9fff\uac00-\ud7af]",
        response_text,
    )
    letters = sum(character.isalpha() for character in response_text)
    if len(suspicious) >= 6 and len(suspicious) / max(letters, 1) >= 0.05:
        return "garbled mixed-script response"
    return None


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
        self,
        method: str,
        path: str,
        raw_body: bytes,
        headers: dict[str, str],
        timeout: int | None = None,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        connection = http.client.HTTPConnection(
            "127.0.0.1", ROUTER_PORT, timeout=timeout or MODEL_ATTEMPT_TIMEOUT
        )
        try:
            connection.request(
                method, path, body=raw_body or None, headers=headers
            )
            response = connection.getresponse()
            return response.status, response.getheaders(), response.read()
        finally:
            connection.close()

    def _model_request(
        self,
        model: str,
        path: str,
        raw_body: bytes,
        headers: dict[str, str],
        timeout: int,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        provider, separator, upstream_model = model.partition("/")
        destination = DIRECT_PROVIDERS.get(provider)
        if not separator or destination is None:
            return self._router_request(
                "POST", path, raw_body, headers, timeout=timeout
            )

        host, provider_path, key_env = destination
        # PowerShell 5.1 pipelines can leave CR/LF characters in a secret
        # transferred over SSH. API keys never contain whitespace, so remove
        # it defensively before constructing an HTTP header.
        api_key = "".join(os.environ.get(key_env, "").split())
        if not api_key:
            body = json.dumps(
                {"error": {"message": f"Missing {key_env}"}}
            ).encode()
            return 503, [("Content-Type", "application/json")], body

        provider_payload = json.loads(raw_body)
        provider_payload["model"] = upstream_model
        provider_body = json.dumps(
            provider_payload, ensure_ascii=False
        ).encode("utf-8")
        provider_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # GoRouter's edge currently rejects non-browser user agents with
            # HTTP 403 even when the API key is valid. Scope the compatibility
            # header to that provider; keep an explicit agent identity for the
            # other APIs.
            "User-Agent": (
                "Mozilla/5.0" if provider == "gorouter"
                else "hermes-text-gateway/1.0"
            ),
            "Content-Length": str(len(provider_body)),
        }
        connection = http.client.HTTPSConnection(host, timeout=timeout)
        try:
            connection.request(
                "POST", provider_path, body=provider_body, headers=provider_headers
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

    def _send_completed_json(self, status: int, payload: dict[str, Any]) -> None:
        """Return a buffered completion to clients that did not request SSE."""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _fallback_completion(
        self,
        request_payload: dict[str, Any],
        forwarded_headers: dict[str, str],
        stream_response: bool,
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
            response_headers: list[tuple[str, str]] = []
            response_body = b""
            status: int | None = None
            request_failed = False
            rate_limit_retries = 0
            tool_repair_used = False
            while True:
                try:
                    status, response_headers, response_body = self._model_request(
                        model,
                        self.path,
                        attempt_body,
                        attempt_headers,
                        timeout=attempt_timeout_for_model(model),
                    )
                except (ConnectionError, OSError, TimeoutError, ValueError, socket.timeout,
                        http.client.HTTPException) as error:
                    failures.append(f"{model}: {type(error).__name__}")
                    cool_down_model(model, None)
                    print(
                        f"[model-proxy] Model {model} failed: {type(error).__name__}",
                        flush=True,
                    )
                    request_failed = True
                    break

                if status == 400 and not tool_repair_used and tool_call_failed(
                    status, response_body
                ):
                    tool_repair_used = True
                    attempt_payload = add_repair_instruction(
                        attempt_payload,
                        "Your previous tool call was rejected. If a tool is needed, "
                        "emit exactly one valid tool call whose arguments are strict "
                        "JSON matching the supplied schema. Otherwise answer in text.",
                    )
                    attempt_body = json.dumps(
                        attempt_payload, ensure_ascii=False
                    ).encode("utf-8")
                    attempt_headers["Content-Length"] = str(len(attempt_body))
                    print(
                        f"[model-proxy] {model} produced an invalid tool call; "
                        "retrying once with JSON repair guidance",
                        flush=True,
                    )
                    continue

                if status != 429 or rate_limit_retries >= RATE_LIMIT_RETRIES:
                    break
                retry_delay = rate_limit_retry_seconds(
                    response_headers, response_body
                )
                if retry_delay is None or retry_delay > RATE_LIMIT_MAX_WAIT:
                    break
                retry_delay = max(0.5, retry_delay + 0.75)
                print(
                    f"[model-proxy] {model} rate-limited; retrying in "
                    f"{retry_delay:.2f}s ({rate_limit_retries + 1}/{RATE_LIMIT_RETRIES})",
                    flush=True,
                )
                time.sleep(retry_delay)
                rate_limit_retries += 1

            if request_failed or status is None:
                continue

            if 200 <= status < 300:
                try:
                    completed = json.loads(response_body)
                    quality_error = response_quality_error(
                        completed, request_payload
                    )
                    if quality_error:
                        raise ValueError(quality_error)
                    # Validate buffered provider output for both transports.
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
                if stream_response:
                    self._send_stream(status, stream_body)
                else:
                    self._send_completed_json(status, completed)
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
                and request_payload is not None
                and request_payload.get("model") == COMBO_MODEL
            ):
                # Always resolve the managed combo here. Previously only SSE
                # requests took this path, so the gateway's non-streaming mode
                # leaked the combo back to 9Router. 9Router then treated the
                # GoRouter credential as an OpenAI key and returned a 401.
                self._fallback_completion(
                    request_payload,
                    forwarded_headers,
                    stream_response=requested_stream,
                )
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
