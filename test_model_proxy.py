import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from openai import OpenAI

import model_proxy
from model_proxy import response_to_sse


def _events(body: bytes):
    return [
        line.removeprefix("data: ")
        for line in body.decode("utf-8").splitlines()
        if line.startswith("data: ")
    ]


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


def test_http_adapter_buffers_router_and_streams_to_openai_sdk(monkeypatch):
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

    router = ThreadingHTTPServer(("127.0.0.1", 0), FakeRouter)
    monkeypatch.setattr(model_proxy, "ROUTER_PORT", router.server_port)
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

    assert response.status_code == 429
    assert response.json()["error"]["message"] == "rate limited"
