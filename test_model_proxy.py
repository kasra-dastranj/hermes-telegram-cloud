import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from openai import OpenAI

import model_proxy
from model_proxy import (
    bounded_messages,
    compact_tools,
    model_cooldown_seconds,
    rate_limit_retry_seconds,
    request_for_model,
    response_to_sse,
    tool_call_failed,
)


def test_compact_tools_preserves_schema_and_removes_documentation_bulk():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "x" * 500,
                "parameters": {
                    "title": "Terminal arguments",
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "y" * 1000,
                            "examples": ["echo ok"],
                        }
                    },
                    "required": ["command"],
                },
            },
        }
    ]

    compacted = compact_tools(tools)

    function = compacted[0]["function"]
    assert function["name"] == "terminal"
    assert len(function["description"]) == 240
    assert function["parameters"]["type"] == "object"
    assert function["parameters"]["required"] == ["command"]
    assert "description" not in function["parameters"]["properties"]["command"]
    assert "examples" not in function["parameters"]["properties"]["command"]


def test_groq_attempt_caps_completion_and_compacts_tools():
    source = {
        "model": "hermes-free",
        "messages": [{"role": "user", "content": "سلام"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "x" * 1000,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "max_tokens": 2048,
    }

    attempt = request_for_model(source, "groq/openai/gpt-oss-120b")

    assert attempt["max_tokens"] == 1024
    assert attempt["reasoning_effort"] == "low"
    assert len(attempt["tools"][0]["function"]["description"]) == 240
    assert source["max_tokens"] == 2048
    assert len(source["tools"][0]["function"]["description"]) == 1000


def _events(body: bytes):
    return [
        line.removeprefix("data: ")
        for line in body.decode("utf-8").splitlines()
        if line.startswith("data: ")
    ]


def test_groq_attempt_removes_unsupported_reasoning_history_without_mutation():
    source = {
        "model": "hermes-free",
        "messages": [
            {"role": "user", "content": "سلام"},
            {
                "role": "assistant",
                "content": "پاسخ",
                "reasoning_details": [{"type": "summary", "text": "private"}],
                "reasoning_content": "private",
                "tool_calls": [{"id": "call_1"}],
            },
        ],
        "stream": True,
    }

    groq = request_for_model(source, "groq/openai/gpt-oss-120b")
    opencode = request_for_model(source, "oc/nemotron-3-ultra-free")

    assert groq["model"] == "groq/openai/gpt-oss-120b"
    assert groq["stream"] is False
    assert groq["messages"][1] == {
        "role": "assistant",
        "content": "پاسخ",
        "tool_calls": [{"id": "call_1"}],
    }
    assert "reasoning_details" in opencode["messages"][1]
    assert "reasoning_details" in source["messages"][1]


def test_history_budget_keeps_system_and_recent_messages_and_trims_tool_output():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old" * 3000},
        {"role": "assistant", "content": "working"},
        {"role": "tool", "content": "x" * 12000},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "latest"},
    ]

    bounded = bounded_messages(messages, 7000)

    assert bounded[0] == {"role": "system", "content": "rules"}
    assert bounded[-1] == {"role": "user", "content": "latest"}
    assert all(message.get("content") != "old" * 3000 for message in bounded)
    tool_messages = [message for message in bounded if message.get("role") == "tool"]
    if tool_messages:
        assert "oversized content trimmed" in tool_messages[0]["content"]


def test_provider_failures_receive_useful_cooldowns():
    assert model_cooldown_seconds(400) == 30
    assert model_cooldown_seconds(401) == 3600
    assert model_cooldown_seconds(429) == 15
    assert model_cooldown_seconds(502) == 300
    assert model_cooldown_seconds(None) == 300


def test_rate_limit_retry_delay_reads_json_message_and_header():
    body = b'{"error":{"message":"Please try again in 5.1825s"}}'
    assert rate_limit_retry_seconds([], body) == pytest.approx(5.1825)
    assert rate_limit_retry_seconds([("Retry-After", "3")], body) == 3
    assert rate_limit_retry_seconds([], b"try again later") is None


def test_tool_call_failure_detection_is_specific():
    assert tool_call_failed(400, b'{"code":"tool_use_failed"}') is True
    assert tool_call_failed(400, b'{"message":"tool call validation failed"}') is True
    assert tool_call_failed(429, b'{"code":"tool_use_failed"}') is False


def test_direct_groq_request_strips_provider_prefix_and_uses_secret(monkeypatch):
    captured = {}

    class FakeResponse:
        status = 200

        @staticmethod
        def getheaders():
            return [("Content-Type", "application/json")]

        @staticmethod
        def read():
            return b'{"choices":[]}'

    class FakeConnection:
        def __init__(self, host, timeout):
            captured["host"] = host
            captured["timeout"] = timeout

        def request(self, method, path, body, headers):
            captured.update(
                method=method,
                path=path,
                payload=json.loads(body),
                authorization=headers["Authorization"],
            )

        @staticmethod
        def getresponse():
            return FakeResponse()

        @staticmethod
        def close():
            pass

    monkeypatch.setenv("GROQ_API_KEY", "test-secret")
    monkeypatch.setattr(model_proxy.http.client, "HTTPSConnection", FakeConnection)
    handler = object.__new__(model_proxy.Handler)
    status, _headers, _body = handler._model_request(
        "groq/openai/gpt-oss-120b",
        "/v1/chat/completions",
        json.dumps({"model": "groq/openai/gpt-oss-120b"}).encode(),
        {},
        20,
    )

    assert status == 200
    assert captured["host"] == "api.groq.com"
    assert captured["path"] == "/openai/v1/chat/completions"
    assert captured["payload"]["model"] == "openai/gpt-oss-120b"
    assert captured["authorization"] == "Bearer test-secret"


def test_response_to_sse_preserves_text_and_finish_reason():
    body = response_to_sse(
        {
            "id": "abc",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "سلام"},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    events = _events(body)

    first = json.loads(events[0])
    final = json.loads(events[1])
    assert first["choices"][0]["delta"]["content"] == "سلام"
    assert final["choices"][0]["finish_reason"] == "stop"
    assert events[-1] == "[DONE]"


def test_response_to_sse_adds_tool_call_indexes():
    body = response_to_sse(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "cronjob",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    first = json.loads(_events(body)[0])

    assert first["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert json.loads(_events(body)[1])["choices"][0]["finish_reason"] == "tool_calls"


def test_response_to_sse_rejects_success_without_choices():
    with pytest.raises(ValueError, match="no choices"):
        response_to_sse({"error": {"message": "provider failed"}})


def test_response_to_sse_rejects_empty_assistant_message():
    with pytest.raises(ValueError, match="no usable output"):
        response_to_sse(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "stop",
                    }
                ]
            }
        )


def test_http_adapter_buffers_router_and_streams_to_openai_sdk(
    monkeypatch, tmp_path
):
    received = {}

    class FakeRouter(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            received.update(json.loads(self.rfile.read(length)))
            body = json.dumps(
                {
                    "id": "chatcmpl-integration",
                    "model": "fallback-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "پاسخ یکپارچه سالم است",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    manifest = tmp_path / "fallback-models.json"
    manifest.write_text(json.dumps(["provider/test"]), encoding="utf-8")
    router = ThreadingHTTPServer(("127.0.0.1", 0), FakeRouter)
    monkeypatch.setattr(model_proxy, "ROUTER_PORT", router.server_port)
    monkeypatch.setattr(model_proxy, "FALLBACK_MODELS_FILE", str(manifest))
    adapter = ThreadingHTTPServer(("127.0.0.1", 0), model_proxy.Handler)
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in (router, adapter)
    ]
    for thread in threads:
        thread.start()

    try:
        client = OpenAI(
            base_url=f"http://127.0.0.1:{adapter.server_port}/v1",
            api_key="local-test",
        )
        chunks = client.chat.completions.create(
            model="hermes-free",
            messages=[{"role": "user", "content": "سلام"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        text = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in chunks
            if chunk.choices
        )
    finally:
        adapter.shutdown()
        router.shutdown()
        adapter.server_close()
        router.server_close()

    assert text == "پاسخ یکپارچه سالم است"
    assert received["stream"] is False
    assert "stream_options" not in received


def test_http_adapter_preserves_router_error(monkeypatch):
    class RateLimitedRouter(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = b'{"error":{"message":"rate limited"}}'
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    router = ThreadingHTTPServer(("127.0.0.1", 0), RateLimitedRouter)
    monkeypatch.setattr(model_proxy, "ROUTER_PORT", router.server_port)
    adapter = ThreadingHTTPServer(("127.0.0.1", 0), model_proxy.Handler)
    for server in (router, adapter):
        threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        response = httpx.post(
            f"http://127.0.0.1:{adapter.server_port}/v1/chat/completions",
            json={
                "model": "direct-test",
                "messages": [{"role": "user", "content": "سلام"}],
                "stream": True,
            },
        )
    finally:
        adapter.shutdown()
        router.shutdown()
        adapter.server_close()
        router.server_close()

    assert response.status_code == 429
    assert response.json()["error"]["message"] == "rate limited"


def test_combo_falls_through_empty_success_to_next_model(monkeypatch, tmp_path):
    attempted_models = []

    class FakeRouter(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            attempted_models.append(request["model"])
            if request["model"] == "provider/empty":
                payload = {
                    "id": "empty",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "stop",
                        }
                    ],
                }
            else:
                payload = {
                    "id": "fallback",
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "مدل دوم پاسخ داد",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    manifest = tmp_path / "fallback-models.json"
    manifest.write_text(
        json.dumps(["provider/empty", "provider/working"]), encoding="utf-8"
    )
    router = ThreadingHTTPServer(("127.0.0.1", 0), FakeRouter)
    monkeypatch.setattr(model_proxy, "ROUTER_PORT", router.server_port)
    monkeypatch.setattr(model_proxy, "FALLBACK_MODELS_FILE", str(manifest))
    adapter = ThreadingHTTPServer(("127.0.0.1", 0), model_proxy.Handler)
    for server in (router, adapter):
        threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        client = OpenAI(
            base_url=f"http://127.0.0.1:{adapter.server_port}/v1",
            api_key="local-test",
        )
        chunks = client.chat.completions.create(
            model="hermes-free",
            messages=[{"role": "user", "content": "تست fallback"}],
            stream=True,
        )
        text = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in chunks
            if chunk.choices
        )
    finally:
        adapter.shutdown()
        router.shutdown()
        adapter.server_close()
        router.server_close()

    assert attempted_models == ["provider/empty", "provider/working"]
    assert text == "مدل دوم پاسخ داد"


def test_combo_returns_json_error_after_all_models_fail(monkeypatch, tmp_path):
    class EmptyRouter(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = b'{"choices":[]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    manifest = tmp_path / "fallback-models.json"
    manifest.write_text(json.dumps(["provider/empty"]), encoding="utf-8")
    router = ThreadingHTTPServer(("127.0.0.1", 0), EmptyRouter)
    monkeypatch.setattr(model_proxy, "ROUTER_PORT", router.server_port)
    monkeypatch.setattr(model_proxy, "FALLBACK_MODELS_FILE", str(manifest))
    adapter = ThreadingHTTPServer(("127.0.0.1", 0), model_proxy.Handler)
    for server in (router, adapter):
        threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        response = httpx.post(
            f"http://127.0.0.1:{adapter.server_port}/v1/chat/completions",
            json={
                "model": "hermes-free",
                "messages": [{"role": "user", "content": "سلام"}],
                "stream": True,
            },
        )
    finally:
        adapter.shutdown()
        router.shutdown()
        adapter.server_close()
        router.server_close()

    assert response.status_code == 502
    assert response.headers["content-type"] == "application/json"
    assert "All 1 fallback models failed" in response.json()["error"]["message"]


def test_combo_falls_through_timeout_to_next_model(monkeypatch, tmp_path):
    attempted_models = []

    def fake_router_request(
        _self, _method, _path, raw_body, _headers, timeout=None
    ):
        model = json.loads(raw_body)["model"]
        attempted_models.append(model)
        if model == "provider/timeout":
            raise socket.timeout("simulated timeout")
        body = json.dumps(
            {
                "model": model,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "بازیابی پس از timeout",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        return 200, [("Content-Type", "application/json")], body

    manifest = tmp_path / "fallback-models.json"
    manifest.write_text(
        json.dumps(["provider/timeout", "provider/working"]), encoding="utf-8"
    )
    monkeypatch.setattr(model_proxy, "FALLBACK_MODELS_FILE", str(manifest))
    monkeypatch.setattr(model_proxy.Handler, "_router_request", fake_router_request)
    adapter = ThreadingHTTPServer(("127.0.0.1", 0), model_proxy.Handler)
    threading.Thread(target=adapter.serve_forever, daemon=True).start()

    try:
        response = httpx.post(
            f"http://127.0.0.1:{adapter.server_port}/v1/chat/completions",
            json={
                "model": "hermes-free",
                "messages": [{"role": "user", "content": "سلام"}],
                "stream": True,
            },
        )
    finally:
        adapter.shutdown()
        adapter.server_close()

    assert response.status_code == 200
    assert attempted_models == ["provider/timeout", "provider/working"]
    assert "بازیابی پس از timeout" in response.text


def test_combo_waits_for_short_rate_limit_then_retries_same_model(
    monkeypatch, tmp_path
):
    attempts = []
    sleeps = []

    def fake_router_request(
        _self, _method, _path, raw_body, _headers, timeout=None
    ):
        model = json.loads(raw_body)["model"]
        attempts.append(model)
        if len(attempts) == 1:
            return (
                429,
                [],
                b'{"error":{"message":"Please try again in 0.01s"}}',
            )
        body = json.dumps(
            {
                "model": model,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "بازیابی شد"},
                        "finish_reason": "stop",
                    }
                ],
            },
            ensure_ascii=False,
        ).encode()
        return 200, [], body

    manifest = tmp_path / "fallback-models.json"
    manifest.write_text(json.dumps(["provider/test"]), encoding="utf-8")
    monkeypatch.setattr(model_proxy, "FALLBACK_MODELS_FILE", str(manifest))
    monkeypatch.setattr(model_proxy.Handler, "_router_request", fake_router_request)
    monkeypatch.setattr(model_proxy.time, "sleep", sleeps.append)
    adapter = ThreadingHTTPServer(("127.0.0.1", 0), model_proxy.Handler)
    threading.Thread(target=adapter.serve_forever, daemon=True).start()

    try:
        response = httpx.post(
            f"http://127.0.0.1:{adapter.server_port}/v1/chat/completions",
            json={
                "model": "hermes-free",
                "messages": [{"role": "user", "content": "سلام"}],
                "stream": True,
            },
        )
    finally:
        adapter.shutdown()
        adapter.server_close()

    assert response.status_code == 200
    assert attempts == ["provider/test", "provider/test"]
    assert sleeps == [pytest.approx(0.76)]
    assert "بازیابی شد" in response.text


def test_combo_repairs_one_invalid_tool_call_before_fallback(monkeypatch, tmp_path):
    requests = []

    def fake_router_request(
        _self, _method, _path, raw_body, _headers, timeout=None
    ):
        request = json.loads(raw_body)
        requests.append(request)
        if len(requests) == 1:
            return 400, [], b'{"error":{"code":"tool_use_failed"}}'
        body = json.dumps(
            {
                "model": request["model"],
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "فراخوانی ابزار اصلاح شد",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            ensure_ascii=False,
        ).encode()
        return 200, [], body

    manifest = tmp_path / "fallback-models.json"
    manifest.write_text(json.dumps(["provider/tool-model"]), encoding="utf-8")
    monkeypatch.setattr(model_proxy, "FALLBACK_MODELS_FILE", str(manifest))
    monkeypatch.setattr(model_proxy.Handler, "_router_request", fake_router_request)
    adapter = ThreadingHTTPServer(("127.0.0.1", 0), model_proxy.Handler)
    threading.Thread(target=adapter.serve_forever, daemon=True).start()

    try:
        response = httpx.post(
            f"http://127.0.0.1:{adapter.server_port}/v1/chat/completions",
            json={
                "model": "hermes-free",
                "messages": [{"role": "user", "content": "کرون بساز"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "cronjob",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "stream": True,
            },
        )
    finally:
        adapter.shutdown()
        adapter.server_close()

    assert response.status_code == 200
    assert len(requests) == 2
    repair_messages = [
        message
        for message in requests[1]["messages"]
        if message.get("role") == "system"
        and "strict JSON" in message.get("content", "")
    ]
    assert len(repair_messages) == 1
    assert "فراخوانی ابزار اصلاح شد" in response.text
