import gc
import weakref
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from openai import BadRequestError
from pydantic import BaseModel

from src.exceptions import ValidationException
from src.llm.backends.openai import (
    OpenAIBackend,
    _json_object_instruction,  # pyright: ignore[reportPrivateUsage]
    extract_openai_cache_tokens,
)
from src.utils.representation import PromptRepresentation


class StructuredResponse(BaseModel):
    answer: str


AGENT_TOOL = {
    "name": "search",
    "description": "Search for information",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


def _await_kwargs(mock_method: Any) -> dict[str, Any]:
    call = mock_method.await_args
    if call is None:
        raise AssertionError("Expected mocked method to be awaited")
    return call.kwargs


def _response(
    text: str = "",
    *,
    output: list[Any] | None = None,
    status: str = "completed",
    incomplete_reason: str | None = None,
    cached_tokens: int = 4,
) -> SimpleNamespace:
    return SimpleNamespace(
        output_text=text,
        output=output or [],
        output_parsed=None,
        status=status,
        incomplete_details=(
            SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None
        ),
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=7,
            input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
    )


def _bad_request_error() -> BadRequestError:
    request = httpx.Request("POST", "https://example.test/v1/responses")
    response = httpx.Response(400, request=request)
    return BadRequestError("json_schema unsupported", response=response, body=None)


@pytest.mark.asyncio
async def test_plain_text_uses_responses_and_normalizes_usage_and_reasoning() -> None:
    reasoning = SimpleNamespace(
        type="reasoning",
        summary=[SimpleNamespace(type="summary_text", text="reasoning summary")],
        model_dump=lambda: {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "reasoning summary"}],
        },
    )
    client = Mock()
    client.responses.create = AsyncMock(
        return_value=_response("Hello from Responses", output=[reasoning])
    )

    result = await OpenAIBackend(client).complete(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        thinking_effort="high",
    )

    call = _await_kwargs(client.responses.create)
    assert call["model"] == "gpt-5.6-luna"
    assert call["input"] == [{"role": "user", "content": "Hello"}]
    assert call["max_output_tokens"] == 100
    assert call["store"] is False
    assert call["reasoning"] == {"effort": "high"}
    assert call["include"] == ["reasoning.encrypted_content"]
    assert result.content == "Hello from Responses"
    assert result.thinking_content == "reasoning summary"
    assert result.reasoning_details[0]["id"] == "rs_1"
    assert result.input_tokens == 12
    assert result.output_tokens == 7
    assert result.cache_read_input_tokens == 4
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_all_model_names_use_responses_api() -> None:
    client = Mock()
    client.responses.create = AsyncMock(return_value=_response("ok"))

    result = await OpenAIBackend(client).complete(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=50,
    )

    assert result.content == "ok"
    client.responses.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_max_output_override_and_supported_provider_params() -> None:
    client = Mock()
    client.responses.create = AsyncMock(return_value=_response())

    await OpenAIBackend(client).complete(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        max_output_tokens=42,
        temperature=0.2,
        extra_params={
            "top_p": 0.8,
            "verbosity": "low",
            "extra_headers": {"X-Test": "yes"},
            "extra_query": {"trace": "abc"},
            "extra_body": {"custom": True},
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "seed": 7,
        },
    )

    call = _await_kwargs(client.responses.create)
    assert call["max_output_tokens"] == 42
    assert call["temperature"] == 0.2
    assert call["top_p"] == 0.8
    assert call["text"]["verbosity"] == "low"
    assert call["extra_headers"] == {"X-Test": "yes"}
    assert call["extra_query"] == {"trace": "abc"}
    assert call["extra_body"] == {"custom": True}
    assert "frequency_penalty" not in call
    assert "presence_penalty" not in call
    assert "seed" not in call


@pytest.mark.asyncio
async def test_openai_backend_passes_timeout_to_responses_request() -> None:
    """Responses requests receive the upstream per-request provider timeout."""
    client = Mock()
    client.responses.create = AsyncMock(return_value=_response("ok"))

    await OpenAIBackend(client).complete(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        extra_params={"timeout": "30"},
    )

    assert _await_kwargs(client.responses.create)["timeout"] == 30.0


@pytest.mark.asyncio
async def test_pydantic_structured_output_uses_responses_create() -> None:
    response = _response('{"answer":"ok"}')
    client = Mock()
    client.responses.create = AsyncMock(return_value=response)

    result = await OpenAIBackend(client).complete(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "Return JSON"}],
        max_tokens=100,
        response_format=StructuredResponse,
    )

    call = _await_kwargs(client.responses.create)
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["text"]["format"]["strict"] is True
    assert isinstance(result.content, StructuredResponse)
    assert result.content.answer == "ok"


@pytest.mark.asyncio
async def test_structured_output_with_tools_uses_create_and_manual_validation() -> None:
    client = Mock()
    client.responses.create = AsyncMock(return_value=_response('{"answer":"done"}'))

    result = await OpenAIBackend(client).complete(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "Use a tool if needed"}],
        max_tokens=100,
        tools=[AGENT_TOOL],
        tool_choice="auto",
        response_format=StructuredResponse,
    )

    call = _await_kwargs(client.responses.create)
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["text"]["format"]["name"] == "StructuredResponse"
    assert isinstance(result.content, StructuredResponse)
    assert result.content.answer == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("required", "required"),
        ("any", "required"),
        ("auto", "auto"),
        ("none", "none"),
        ("search", {"type": "function", "name": "search"}),
        (
            {"type": "function", "function": {"name": "search"}},
            {"type": "function", "name": "search"},
        ),
    ],
)
async def test_function_tools_and_tool_choices(
    choice: str | dict[str, Any], expected: str | dict[str, Any]
) -> None:
    call_item = SimpleNamespace(
        type="function_call",
        call_id="call_1",
        name="search",
        arguments='{"query":"new"}',
    )
    client = Mock()
    client.responses.create = AsyncMock(return_value=_response(output=[call_item]))

    result = await OpenAIBackend(client).complete(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "Search"}],
        max_tokens=100,
        tools=[AGENT_TOOL],
        tool_choice=choice,
    )

    call = _await_kwargs(client.responses.create)
    assert call["tools"] == [
        {
            "type": "function",
            "name": "search",
            "description": "Search for information",
            "parameters": AGENT_TOOL["input_schema"],
        }
    ]
    assert call["tool_choice"] == expected
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].input == {"query": "new"}


@pytest.mark.asyncio
async def test_malformed_function_arguments_are_nonfatal() -> None:
    item = SimpleNamespace(
        type="function_call", call_id="call_bad", name="search", arguments="{bad"
    )
    client = Mock()
    client.responses.create = AsyncMock(return_value=_response(output=[item]))

    result = await OpenAIBackend(client).complete(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "Search"}],
        max_tokens=100,
        tools=[AGENT_TOOL],
    )

    assert result.tool_calls[0].input == {}


@pytest.mark.asyncio
async def test_chat_shaped_tool_and_reasoning_history_is_converted() -> None:
    client = Mock()
    client.responses.create = AsyncMock(return_value=_response("done"))
    messages = [
        {"role": "user", "content": "Search"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_details": [{"type": "reasoning", "id": "rs_1", "summary": []}],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query":"old"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "old result"},
    ]

    await OpenAIBackend(client).complete(
        model="gpt-5.6-luna", messages=messages, max_tokens=100
    )

    call = _await_kwargs(client.responses.create)
    assert call["input"] == [
        {"role": "user", "content": "Search"},
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "search",
            "arguments": '{"query":"old"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "old result"},
    ]


@pytest.mark.asyncio
async def test_json_object_mode_injects_schema_and_repairs_content() -> None:
    client = Mock()
    client.responses.create = AsyncMock(return_value=_response('{"answer":"ok"}'))

    result = await OpenAIBackend(client).complete(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=StructuredResponse,
        extra_params={"structured_output_mode": "json_object"},
    )

    call = _await_kwargs(client.responses.create)
    assert call["input"][0]["role"] == "system"
    assert "JSON schema" in call["input"][0]["content"]
    assert call["text"]["format"] == {"type": "json_object"}
    assert isinstance(result.content, StructuredResponse)
    assert result.content.answer == "ok"


@pytest.mark.asyncio
async def test_contentless_json_object_returns_empty_prompt_representation() -> None:
    client = Mock()
    client.responses.create = AsyncMock(return_value=_response(""))

    result = await OpenAIBackend(client).complete(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=PromptRepresentation,
        extra_params={"structured_output_mode": "json_object"},
    )

    assert isinstance(result.content, PromptRepresentation)


@pytest.mark.asyncio
async def test_missing_structured_content_raises() -> None:
    client = Mock()
    response = _response("")
    client.responses.create = AsyncMock(return_value=response)

    with pytest.raises(ValidationException):
        await OpenAIBackend(client).complete(
            model="gpt-5.6-luna",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            response_format=StructuredResponse,
        )


@pytest.mark.asyncio
async def test_structured_bad_request_returns_safe_fallback() -> None:
    client = Mock()
    client.responses.create = AsyncMock(side_effect=_bad_request_error())

    result = await OpenAIBackend(client).complete(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=StructuredResponse,
    )

    assert result.content == ""


@pytest.mark.asyncio
async def test_dict_response_format_is_mapped_to_responses_text_config() -> None:
    client = Mock()
    client.responses.create = AsyncMock(return_value=_response("{}"))
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {"type": "object"},
            "strict": True,
        },
    }

    await OpenAIBackend(client).complete(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=response_format,
    )

    assert _await_kwargs(client.responses.create)["text"]["format"] == {
        "type": "json_schema",
        "name": "answer",
        "schema": {"type": "object"},
        "strict": True,
    }


@pytest.mark.asyncio
async def test_incomplete_response_maps_finish_reason() -> None:
    client = Mock()
    client.responses.create = AsyncMock(
        return_value=_response(
            "partial", status="incomplete", incomplete_reason="max_output_tokens"
        )
    )

    result = await OpenAIBackend(client).complete(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10,
    )

    assert result.finish_reason == "length"


class EventStream:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.index = 0

    def __aiter__(self) -> "EventStream":
        return self

    async def __anext__(self) -> Any:
        if self.index >= len(self.events):
            raise StopAsyncIteration
        event = self.events[self.index]
        self.index += 1
        return event


@pytest.mark.asyncio
async def test_stream_uses_responses_events_for_text_and_final_usage() -> None:
    final = _response("hello")
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="hel"),
        SimpleNamespace(type="response.output_text.delta", delta="lo"),
        SimpleNamespace(type="response.completed", response=final),
    ]
    client = Mock()
    client.responses.create = AsyncMock(return_value=EventStream(events))

    chunks = [
        chunk
        async for chunk in OpenAIBackend(client).stream(
            model="gpt-5.6-luna",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )
    ]

    call = _await_kwargs(client.responses.create)
    assert call["stream"] is True
    assert [chunk.content for chunk in chunks[:-1]] == ["hel", "lo"]
    assert chunks[-1].is_done is True
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].output_tokens == 7


def test_cache_extraction_supports_responses_and_empty_usage() -> None:
    assert extract_openai_cache_tokens(None) == (0, 0)
    usage = SimpleNamespace(
        input_tokens_details=SimpleNamespace(cached_tokens=9),
        cache_creation_input_tokens=3,
    )
    assert extract_openai_cache_tokens(usage) == (3, 9)


def test_json_instruction_cache_does_not_retain_dynamic_models() -> None:
    model = type("DynamicResponse", (BaseModel,), {"__annotations__": {"value": str}})
    reference = weakref.ref(model)
    assert "JSON schema" in _json_object_instruction(model)
    del model
    gc.collect()
    assert reference() is None
