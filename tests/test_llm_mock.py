"""Behavioural tests for :class:`clyde.llm.MockLLMClient`.

These tests pin down the mock's contract: queue vs router precedence,
queue-exhaustion error messages, call-log fidelity, JSON coercion, and
Protocol runtime-checkability.
"""

from __future__ import annotations

import json

import pytest

from clyde.llm import LLMClient, LLMMessage, LLMResponse, MockLLMClient


# --------------------------------------------------------- dataclass fields


def test_llm_message_fields_populate():
    msg = LLMMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_llm_response_fields_populate_with_defaults():
    resp = LLMResponse(content="hi", model="mock-1")
    assert resp.content == "hi"
    assert resp.model == "mock-1"
    assert resp.usage == {}
    assert resp.raw is None


def test_llm_response_usage_and_raw_are_settable():
    resp = LLMResponse(
        content="hi",
        model="mock-1",
        usage={"input_tokens": 3, "output_tokens": 5},
        raw={"id": "abc"},
    )
    assert resp.usage == {"input_tokens": 3, "output_tokens": 5}
    assert resp.raw == {"id": "abc"}


def test_llm_response_usage_default_is_per_instance():
    """``field(default_factory=dict)`` must not alias across instances."""
    a = LLMResponse(content="a", model="m")
    b = LLMResponse(content="b", model="m")
    a.usage["x"] = 1
    assert b.usage == {}


# ---------------------------------------------------- Protocol conformance


def test_mock_client_is_runtime_llm_client():
    client = MockLLMClient()
    assert isinstance(client, LLMClient)


# ------------------------------------------------------------ queue mode


@pytest.mark.asyncio
async def test_queue_dequeues_in_fifo_order_across_both_methods():
    client = MockLLMClient(
        responses=[
            LLMResponse(content="first", model="mock-1"),
            {"second": True},
            "third",
        ]
    )

    r1 = await client.complete([LLMMessage(role="user", content="a")])
    assert r1.content == "first"
    assert r1.model == "mock-1"

    r2 = await client.complete_json([LLMMessage(role="user", content="b")])
    assert r2 == {"second": True}

    r3 = await client.complete([LLMMessage(role="user", content="c")])
    assert r3.content == "third"
    assert r3.model == "mock-1"  # default model fills in for bare strings


@pytest.mark.asyncio
async def test_queue_dict_entry_serialized_as_json_for_complete():
    client = MockLLMClient(responses=[{"k": 1, "v": [2, 3]}])
    resp = await client.complete([LLMMessage(role="user", content="x")])
    assert isinstance(resp, LLMResponse)
    assert json.loads(resp.content) == {"k": 1, "v": [2, 3]}


@pytest.mark.asyncio
async def test_queue_dict_entry_returned_directly_for_complete_json():
    payload = {"k": 1, "v": [2, 3]}
    client = MockLLMClient(responses=[payload])
    resp = await client.complete_json([LLMMessage(role="user", content="x")])
    assert resp == payload


@pytest.mark.asyncio
async def test_queue_str_entry_wrapped_into_response():
    client = MockLLMClient(responses=["plain text"], default_model="mock-42")
    resp = await client.complete([LLMMessage(role="user", content="x")])
    assert resp.content == "plain text"
    assert resp.model == "mock-42"


@pytest.mark.asyncio
async def test_queue_exhaustion_raises_indexerror_with_clear_message():
    client = MockLLMClient(responses=["one"])
    await client.complete([LLMMessage(role="user", content="x")])
    with pytest.raises(IndexError) as excinfo:
        await client.complete([LLMMessage(role="user", content="y")])
    assert "MockLLMClient" in str(excinfo.value)
    assert "empty" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_enqueue_appends_to_queue():
    client = MockLLMClient()
    client.enqueue("added later")
    resp = await client.complete([LLMMessage(role="user", content="x")])
    assert resp.content == "added later"


# ------------------------------------------------------------ router mode


@pytest.mark.asyncio
async def test_router_routes_based_on_message_content():
    def route(messages):
        last = messages[-1].content
        if "greet" in last:
            return "hello!"
        if "json" in last:
            return {"result": 42}
        return LLMResponse(content="fallback", model="mock-router")

    client = MockLLMClient(router=route)

    r1 = await client.complete([LLMMessage(role="user", content="please greet me")])
    assert r1.content == "hello!"

    r2 = await client.complete_json([LLMMessage(role="user", content="give json")])
    assert r2 == {"result": 42}

    r3 = await client.complete([LLMMessage(role="user", content="anything else")])
    assert r3.content == "fallback"
    assert r3.model == "mock-router"


@pytest.mark.asyncio
async def test_router_takes_precedence_over_queue():
    def always_hi(messages):
        return "from-router"

    client = MockLLMClient(responses=["from-queue"], router=always_hi)
    resp = await client.complete([LLMMessage(role="user", content="x")])
    assert resp.content == "from-router"
    # Queue should be untouched when router handles the call.
    resp2 = await client.complete([LLMMessage(role="user", content="y")])
    assert resp2.content == "from-router"


@pytest.mark.asyncio
async def test_router_dict_return_used_directly_for_complete_json():
    client = MockLLMClient(router=lambda msgs: {"routed": True})
    resp = await client.complete_json([LLMMessage(role="user", content="x")])
    assert resp == {"routed": True}


# ------------------------------------------------------------ call_log


@pytest.mark.asyncio
async def test_call_log_records_method_messages_and_kwargs():
    client = MockLLMClient(responses=["r1", {"k": 1}])
    msgs1 = [LLMMessage(role="user", content="ping")]
    msgs2 = [LLMMessage(role="system", content="be brief"),
             LLMMessage(role="user", content="summarise")]

    await client.complete(msgs1, model="mock-a", temperature=0.2, max_tokens=16)
    await client.complete_json(msgs2, schema={"type": "object"}, temperature=0.5)

    log = client.call_log
    assert len(log) == 2

    m0, messages0, kwargs0 = log[0]
    assert m0 == "complete"
    assert messages0 == msgs1
    assert kwargs0["model"] == "mock-a"
    assert kwargs0["temperature"] == 0.2
    assert kwargs0["max_tokens"] == 16

    m1, messages1, kwargs1 = log[1]
    assert m1 == "complete_json"
    assert messages1 == msgs2
    assert kwargs1["schema"] == {"type": "object"}
    assert kwargs1["temperature"] == 0.5


@pytest.mark.asyncio
async def test_call_log_is_a_copy():
    """Mutating the returned call_log must not affect internal state."""
    client = MockLLMClient(responses=["r"])
    await client.complete([LLMMessage(role="user", content="x")])
    snapshot = client.call_log
    snapshot.clear()
    # The internal log should still hold the recorded call.
    assert len(client.call_log) == 1


# ------------------------------------------------------------ complete_json coercion


@pytest.mark.asyncio
async def test_complete_json_parses_queued_json_string():
    client = MockLLMClient(responses=['{"parsed": true}'])
    resp = await client.complete_json([LLMMessage(role="user", content="x")])
    assert resp == {"parsed": True}


@pytest.mark.asyncio
async def test_complete_json_parses_queued_llmresponse_with_json_content():
    client = MockLLMClient(
        responses=[LLMResponse(content='{"from": "response"}', model="mock-1")]
    )
    resp = await client.complete_json([LLMMessage(role="user", content="x")])
    assert resp == {"from": "response"}


@pytest.mark.asyncio
async def test_complete_json_rejects_non_json_string():
    client = MockLLMClient(responses=["not json at all"])
    with pytest.raises(ValueError):
        await client.complete_json([LLMMessage(role="user", content="x")])
