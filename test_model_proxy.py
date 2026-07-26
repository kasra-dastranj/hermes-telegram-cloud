import json

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
