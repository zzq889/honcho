import gc
import logging
import weakref
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from openai import BadRequestError
from pydantic import BaseModel

from src.exceptions import ValidationException
from src.llm.structured_output import StructuredOutputError
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
            "reasoning_details": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "status": "completed",
                }
            ],
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
        # ``status`` is output-only in the live API and must not be replayed.
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

    with pytest.raises(StructuredOutputError):
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
    assert "JSON schema" in _json_object_instruction(model, tools_present=False)
    del model
    gc.collect()
    assert reference() is None


# thinking_tool_choice_conflict on the OpenAI transport. An OpenAI-compatible
# proxy in front of an Anthropic model may turn a json_schema response_format
# into a forced tool call, so the operator's policy has to cover schemas as
# well as forced tool choices.


@pytest.mark.asyncio
@pytest.mark.parametrize("with_tools", [False, True])
async def test_json_object_instruction_is_tool_aware(with_tools: bool) -> None:
    client = _conflict_client()
    await OpenAIBackend(client).complete(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        tools=[AGENT_TOOL] if with_tools else None,
        response_format=StructuredResponse,
        extra_params={"structured_output_mode": "json_object"},
    )
    content = _sent_kwargs(client)["input"][0]["content"]
    assert content.startswith(
        "If you are not calling a tool" if with_tools else "You must respond"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text", ["private-sentinel-prose", '{"wrong": "private-sentinel"}']
)
async def test_malformed_representation_is_safely_rejected(text: str) -> None:
    client = Mock()
    client.responses.create = AsyncMock(return_value=_response(text))
    with pytest.raises(StructuredOutputError) as exc:
        await OpenAIBackend(client).complete(
            model="gpt-6-luna",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            response_format=PromptRepresentation,
        )
    assert "private-sentinel" not in str(exc.value)
    assert "payload_sha256=" in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, expected",
    [("```json\n{}\n```", []), ('{"explicit": ["Likes coffee"]}', ["Likes coffee"])],
)
async def test_representation_repairs_fences_and_string_shapes(
    text: str, expected: list[str]
) -> None:
    client = Mock()
    client.responses.create = AsyncMock(return_value=_response(text))
    result = await OpenAIBackend(client).complete(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=PromptRepresentation,
    )
    assert [item.content for item in result.content.explicit] == expected


CONFLICT_MODEL = "claude-via-proxy"


def _conflict_client() -> Mock:
    client = Mock()
    client.responses.create = AsyncMock(return_value=_response('{"answer":"ok"}'))
    return client


def _sent_kwargs(client: Mock) -> dict[str, Any]:
    return _await_kwargs(client.responses.create)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_tools", [False, True], ids=["parse", "tool_loop"])
async def test_override_thinking_drops_reasoning_effort_for_schema(
    with_tools: bool,
) -> None:
    client = _conflict_client()

    await OpenAIBackend(client).complete(
        model=CONFLICT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        tools=[AGENT_TOOL] if with_tools else None,
        tool_choice="auto" if with_tools else None,
        response_format=StructuredResponse,
        thinking_effort="low",
        extra_params={"thinking_tool_choice_conflict": "override_thinking"},
    )

    call = _sent_kwargs(client)
    assert "reasoning" not in call
    # The schema and the agent's own tool choice are untouched.
    if with_tools:
        assert call["text"]["format"]["type"] == "json_schema"
        assert call["tool_choice"] == "auto"
    else:
        assert call["text"]["format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_override_thinking_strips_thinking_from_extra_body_only() -> None:
    client = _conflict_client()

    await OpenAIBackend(client).complete(
        model=CONFLICT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=StructuredResponse,
        thinking_budget_tokens=2048,
        extra_params={
            "thinking_tool_choice_conflict": "override_thinking",
            "extra_body": {
                "thinking": {"type": "adaptive"},
                "metadata": {"route": "primary"},
            },
        },
    )

    assert _sent_kwargs(client)["extra_body"] == {"metadata": {"route": "primary"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_choice", "sent_tool_choice"), [("any", "required"), ("search", None)]
)
async def test_override_thinking_drops_reasoning_effort_for_forced_tool(
    tool_choice: str, sent_tool_choice: str | None
) -> None:
    client = _conflict_client()

    await OpenAIBackend(client).complete(
        model=CONFLICT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        tools=[AGENT_TOOL],
        tool_choice=tool_choice,
        thinking_effort="low",
        extra_params={"thinking_tool_choice_conflict": "override_thinking"},
    )

    call = _sent_kwargs(client)
    assert "reasoning" not in call
    if sent_tool_choice is not None:
        assert call["tool_choice"] == sent_tool_choice
    else:
        assert call["tool_choice"] == {
            "type": "function",
            "name": "search",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_params",
    [
        None,
        {"thinking_tool_choice_conflict": "override_thinking"},
    ],
    ids=["no_policy", "no_conflict"],
)
async def test_thinking_kept_without_policy_or_conflict(
    extra_params: dict[str, Any] | None,
) -> None:
    """Without a policy, OpenAI reasoning models keep reasoning_effort next to a
    schema. With a policy but no schema or forced tool there is no conflict."""
    client = _conflict_client()

    await OpenAIBackend(client).complete(
        model=CONFLICT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        tools=[AGENT_TOOL],
        tool_choice="auto",
        response_format=StructuredResponse if extra_params is None else None,
        thinking_effort="low",
        extra_params=extra_params,
    )

    assert _sent_kwargs(client)["reasoning"] == {"effort": "low"}


@pytest.mark.asyncio
async def test_override_tool_requests_schema_as_json_object_and_keeps_thinking() -> (
    None
):
    client = _conflict_client()

    result = await OpenAIBackend(client).complete(
        model=CONFLICT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=StructuredResponse,
        thinking_effort="low",
        extra_params={"thinking_tool_choice_conflict": "override_tool"},
    )

    assert isinstance(result.content, StructuredResponse)
    client.responses.parse.assert_not_called()
    call = _sent_kwargs(client)
    assert call["reasoning"] == {"effort": "low"}
    assert call["text"]["format"] == {"type": "json_object"}
    assert "JSON schema" in call["input"][0]["content"]


@pytest.mark.asyncio
async def test_override_tool_relaxes_forced_tool_choice_and_keeps_thinking() -> None:
    client = _conflict_client()

    await OpenAIBackend(client).complete(
        model=CONFLICT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        tools=[AGENT_TOOL],
        tool_choice="any",
        thinking_effort="low",
        extra_params={"thinking_tool_choice_conflict": "override_tool"},
    )

    call = _sent_kwargs(client)
    assert call["reasoning"] == {"effort": "low"}
    assert call["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_throw_raises_before_sending() -> None:
    client = _conflict_client()

    with pytest.raises(ValidationException, match="Cannot send request"):
        await OpenAIBackend(client).complete(
            model=CONFLICT_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            response_format=StructuredResponse,
            thinking_effort="low",
            extra_params={"thinking_tool_choice_conflict": "throw"},
        )

    client.responses.parse.assert_not_called()
    client.responses.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_extra_body_thinking_is_not_a_conflict() -> None:
    client = _conflict_client()

    await OpenAIBackend(client).complete(
        model=CONFLICT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=StructuredResponse,
        extra_params={
            "thinking_tool_choice_conflict": "throw",
            "extra_body": {"thinking": {"type": "disabled"}},
        },
    )

    assert _sent_kwargs(client)["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_stream_override_thinking_drops_reasoning_effort_for_schema() -> None:
    client = Mock()
    client.responses.create = AsyncMock(return_value=EventStream([]))

    async for _ in OpenAIBackend(client).stream(
        model=CONFLICT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=StructuredResponse,
        thinking_effort="low",
        extra_params={"thinking_tool_choice_conflict": "override_thinking"},
    ):
        pass

    call = _await_kwargs(client.responses.create)
    assert "reasoning" not in call
    assert call["text"]["format"]["type"] == "json_schema"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("thinking_kwargs", "extra_body"),
    [
        ({"thinking_budget_tokens": 2048}, None),
        ({}, {"reasoning_effort": "low"}),
        ({}, {"thinking": {"type": "adaptive"}}),
    ],
    ids=["injected_budget", "extra_body_effort", "extra_body_thinking"],
)
async def test_override_thinking_removes_thinking_only_extra_body(
    thinking_kwargs: dict[str, Any], extra_body: dict[str, Any] | None
) -> None:
    client = _conflict_client()
    extra_params: dict[str, Any] = {
        "thinking_tool_choice_conflict": "override_thinking"
    }
    if extra_body is not None:
        extra_params["extra_body"] = extra_body

    await OpenAIBackend(client).complete(
        model=CONFLICT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=StructuredResponse,
        extra_params=extra_params,
        **thinking_kwargs,
    )

    assert "extra_body" not in _sent_kwargs(client)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reasoning",
    [{"enabled": False}, {"effort": "none"}],
)
async def test_disabled_extra_body_reasoning_is_not_a_conflict(
    reasoning: dict[str, Any],
) -> None:
    client = _conflict_client()

    await OpenAIBackend(client).complete(
        model=CONFLICT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=StructuredResponse,
        extra_params={
            "thinking_tool_choice_conflict": "throw",
            "extra_body": {"reasoning": reasoning},
        },
    )

    assert _sent_kwargs(client)["extra_body"] == {"reasoning": reasoning}


@pytest.mark.asyncio
async def test_forced_tool_untouched_without_policy() -> None:
    client = _conflict_client()

    await OpenAIBackend(client).complete(
        model=CONFLICT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        tools=[AGENT_TOOL],
        tool_choice="any",
        thinking_effort="low",
    )

    call = _sent_kwargs(client)
    assert call["reasoning"] == {"effort": "low"}
    assert call["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_unknown_policy_logs_an_error_and_applies_no_policy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Config load rejects unknown modes; one that reaches the backend from a
    ModelConfig built in code is logged and treated as unset, not raised."""
    client = _conflict_client()

    with caplog.at_level(logging.ERROR, logger="src.llm.request_builder"):
        await OpenAIBackend(client).complete(
            model=CONFLICT_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            response_format=StructuredResponse,
            thinking_effort="low",
            extra_params={"thinking_tool_choice_conflict": "bogus"},
        )

    assert _sent_kwargs(client)["reasoning"] == {"effort": "low"}
    assert "Unknown thinking_tool_choice_conflict mode 'bogus'" in caplog.text


@pytest.mark.asyncio
async def test_stream_override_tool_requests_json_object() -> None:
    client = Mock()
    client.responses.create = AsyncMock(return_value=EventStream([]))

    async for _ in OpenAIBackend(client).stream(
        model=CONFLICT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=StructuredResponse,
        thinking_effort="low",
        extra_params={"thinking_tool_choice_conflict": "override_tool"},
    ):
        pass

    call = _await_kwargs(client.responses.create)
    assert call["reasoning"] == {"effort": "low"}
    assert call["text"]["format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_stream_throw_raises_before_sending() -> None:
    client = Mock()
    client.responses.create = AsyncMock(return_value=EventStream([]))

    with pytest.raises(ValidationException, match="Cannot send request"):
        async for _ in OpenAIBackend(client).stream(
            model=CONFLICT_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            response_format=StructuredResponse,
            thinking_effort="low",
            extra_params={"thinking_tool_choice_conflict": "throw"},
        ):
            pass

    client.responses.create.assert_not_awaited()
