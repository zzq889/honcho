from __future__ import annotations

import json
import logging
import weakref
from collections.abc import AsyncIterator
from typing import Any, cast

from openai import BadRequestError
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, ValidationError

from src.exceptions import ValidationException
from src.llm.backend import CompletionResult, StreamChunk, ToolCallResult
from src.llm.request_builder import (
    apply_sdk_passthroughs,
    request_timeout_from_extra_params,
)
from src.llm.structured_output import (
    StructuredOutputError,
    empty_structured_output,
    repair_response_model_json,
    validate_structured_output,
)

logger = logging.getLogger(__name__)

# Dynamic response model classes must remain collectible.
_json_object_instruction_cache: weakref.WeakKeyDictionary[type[BaseModel], str] = (
    weakref.WeakKeyDictionary()
)


def _json_object_instruction(response_format: type[BaseModel]) -> str:
    cached = _json_object_instruction_cache.get(response_format)
    if cached is not None:
        return cached
    instruction = (
        "You must respond with a single JSON object (json) that conforms "
        "exactly to the following JSON schema. Do not include any text, "
        "markdown, or code fences outside the JSON object.\n\nJSON schema:\n"
        f"{json.dumps(response_format.model_json_schema())}"
    )
    _json_object_instruction_cache[response_format] = instruction
    return instruction


def extract_openai_cache_tokens(usage: Any) -> tuple[int, int]:
    """Return cache creation/read tokens across Responses usage variants."""
    if not usage:
        return 0, 0

    cache_read = 0
    details = getattr(usage, "input_tokens_details", None)
    if details:
        cache_read = getattr(details, "cached_tokens", 0) or 0
    if cache_read == 0:
        # Retain compatibility with proxy usage objects returned through the SDK.
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            cache_read = getattr(details, "cached_tokens", 0) or 0
        cache_read = (
            cache_read
            or getattr(usage, "cache_read_input_tokens", 0)
            or getattr(usage, "cached_tokens", 0)
            or 0
        )

    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return cache_creation, cache_read


def extract_openai_reasoning_details(response: Any) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "reasoning":
            continue
        if hasattr(item, "model_dump"):
            dumped = item.model_dump()
            if isinstance(dumped, dict):
                details.append(cast(dict[str, Any], dumped))
        elif isinstance(item, dict):
            details.append(cast(dict[str, Any], item))
    return details


def extract_openai_reasoning_content(response: Any) -> str | None:
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "reasoning":
            continue
        summary = getattr(item, "summary", None) or []
        for summary_part in summary:
            text = (
                summary_part.get("text")
                if isinstance(summary_part, dict)
                else getattr(summary_part, "text", None)
            )
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts) if parts else None


class OpenAIBackend:
    """OpenAI provider backend using the Responses API exclusively."""

    def __init__(self, client: Any) -> None:
        self._client: Any = client

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        thinking_budget_tokens: int | None = None,
        thinking_effort: str | None = None,
        max_output_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> CompletionResult:
        # Responses has no stop-sequence parameter. Keep the ProviderBackend
        # signature stable, but intentionally do not forward it.
        del stop
        effective_max_tokens = max_output_tokens or max_tokens
        json_object_model = (
            response_format
            if isinstance(response_format, type)
            and self._structured_output_mode(extra_params) == "json_object"
            else None
        )
        request_messages = messages
        if json_object_model is not None:
            request_messages = self._with_json_schema_instructions(
                messages, json_object_model
            )

        params = self._build_params(
            model=model,
            messages=request_messages,
            max_tokens=effective_max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            thinking_effort=thinking_effort,
            thinking_budget_tokens=thinking_budget_tokens,
            extra_params=extra_params,
        )

        if isinstance(response_format, type):
            params["text"] = {
                "format": self._responses_json_format(
                    response_format, json_object=json_object_model is not None
                )
            }
        elif response_format is not None:
            params["text"] = {"format": self._convert_response_format(response_format)}
        elif extra_params and extra_params.get("json_mode"):
            params["text"] = {"format": {"type": "json_object"}}

        try:
            response = await self._client.responses.create(**params)
        except BadRequestError:
            if isinstance(response_format, type):
                return self._structured_rejection_result(response_format, model)
            raise
        if isinstance(response_format, type):
            if self._response_has_tool_calls(response):
                return self._normalize_response(response)
            content = self._parse_or_repair_structured_content(
                response,
                response_format,
                model,
                empty_on_missing=(
                    json_object_model is not None
                    or getattr(response, "status", None) == "incomplete"
                ),
            )
            return self._normalize_response(response, content_override=content)
        return self._normalize_response(response)

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        thinking_budget_tokens: int | None = None,
        thinking_effort: str | None = None,
        max_output_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        del stop
        json_object_model = (
            response_format
            if isinstance(response_format, type)
            and self._structured_output_mode(extra_params) == "json_object"
            else None
        )
        request_messages = messages
        if json_object_model is not None:
            request_messages = self._with_json_schema_instructions(
                messages, json_object_model
            )
        params = self._build_params(
            model=model,
            messages=request_messages,
            max_tokens=max_output_tokens or max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            thinking_effort=thinking_effort,
            thinking_budget_tokens=thinking_budget_tokens,
            extra_params=extra_params,
        )
        if isinstance(response_format, type):
            params["text"] = {
                "format": self._responses_json_format(
                    response_format, json_object=json_object_model is not None
                )
            }
        elif response_format is not None:
            params["text"] = {"format": self._convert_response_format(response_format)}
        elif extra_params and extra_params.get("json_mode"):
            params["text"] = {"format": {"type": "json_object"}}
        params["stream"] = True

        response_stream = await self._client.responses.create(**params)
        done = False
        async for event in response_stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield StreamChunk(content=delta)
            elif event_type in {
                "response.completed",
                "response.incomplete",
                "response.failed",
            }:
                response = event.response
                normalized = self._normalize_response(response)
                yield StreamChunk(
                    is_done=True,
                    finish_reason=normalized.finish_reason,
                    output_tokens=normalized.output_tokens,
                )
                done = True
        if not done:
            # A prematurely closed event stream has no trustworthy usage/reason.
            return

    def _build_params(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        thinking_effort: str | None,
        thinking_budget_tokens: int | None,
        extra_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "input": self._convert_messages(messages),
            "max_output_tokens": max_tokens,
            # Responses are stored by default. Honcho replays its own history, so
            # retain Chat Completions' stateless behavior and privacy posture.
            "store": False,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if thinking_effort:
            params["reasoning"] = {"effort": thinking_effort}
            # With storage disabled, encrypted reasoning lets a subsequent tool
            # turn replay the provider's reasoning item without server-side state.
            params["include"] = ["reasoning.encrypted_content"]
        # Not native to Responses, but preserve the existing operator escape hatch.
        if thinking_budget_tokens is not None and thinking_budget_tokens > 0:
            params.setdefault("extra_body", {}).setdefault("reasoning", {})[
                "max_tokens"
            ] = thinking_budget_tokens
        if tools:
            params["tools"] = self._convert_tools(tools)
            converted_choice = self._convert_tool_choice(tool_choice)
            if converted_choice is not None:
                params["tool_choice"] = converted_choice
        if extra_params:
            if "top_p" in extra_params:
                params["top_p"] = extra_params["top_p"]
            if extra_params.get("verbosity"):
                params["text"] = {"verbosity": extra_params["verbosity"]}
            apply_sdk_passthroughs(params, extra_params)

        timeout = request_timeout_from_extra_params(extra_params)
        if timeout is not None:
            params["timeout"] = timeout
        return params

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert existing Chat-Completions-shaped history into Response input."""
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "tool":
                converted.append(
                    {
                        "type": "function_call_output",
                        "call_id": message["tool_call_id"],
                        "output": str(message.get("content", "")),
                    }
                )
                continue
            if role == "assistant":
                for detail in message.get("reasoning_details") or []:
                    if isinstance(detail, dict) and detail.get("type") == "reasoning":
                        converted.append(dict(detail))
                content = message.get("content")
                if content not in (None, ""):
                    converted.append({"role": "assistant", "content": content})
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call.get("function", {})
                    converted.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call["id"],
                            "name": function.get("name", ""),
                            "arguments": function.get("arguments", ""),
                        }
                    )
                if content in (None, "") and not message.get("tool_calls"):
                    converted.append({"role": "assistant", "content": content or ""})
                continue
            converted.append(
                {
                    key: value
                    for key, value in message.items()
                    if key in {"role", "content"}
                }
            )
        return converted

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                function = tool["function"]
                item = {
                    "type": "function",
                    "name": function["name"],
                    "parameters": function.get("parameters", {}),
                }
                if function.get("description") is not None:
                    item["description"] = function["description"]
                if function.get("strict") is not None:
                    item["strict"] = function["strict"]
                converted.append(item)
            else:
                item = {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                }
                if tool.get("strict") is not None:
                    item["strict"] = tool["strict"]
                converted.append(item)
        return converted

    @staticmethod
    def _convert_tool_choice(
        tool_choice: str | dict[str, Any] | None,
    ) -> str | dict[str, Any] | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, dict):
            if "name" in tool_choice:
                return {"type": "function", "name": tool_choice["name"]}
            function = tool_choice.get("function")
            if isinstance(function, dict) and "name" in function:
                return {"type": "function", "name": function["name"]}
            return tool_choice
        if tool_choice in {"any", "required"}:
            return "required"
        if tool_choice in {"auto", "none"}:
            return tool_choice
        return {"type": "function", "name": tool_choice}

    @staticmethod
    def _responses_json_format(
        response_format: type[BaseModel], *, json_object: bool = False
    ) -> dict[str, Any]:
        if json_object:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "name": response_format.__name__,
            "schema": to_strict_json_schema(response_format),
            "strict": True,
        }

    @staticmethod
    def _convert_response_format(response_format: dict[str, Any]) -> dict[str, Any]:
        if response_format.get("type") != "json_schema":
            return dict(response_format)
        schema = response_format.get("json_schema", {})
        return {
            "type": "json_schema",
            "name": schema.get("name", "response"),
            "schema": schema.get("schema", {}),
            **({"strict": schema["strict"]} if "strict" in schema else {}),
            **(
                {"description": schema["description"]}
                if "description" in schema
                else {}
            ),
        }

    def _normalize_response(
        self, response: Any, *, content_override: Any | None = None
    ) -> CompletionResult:
        usage = getattr(response, "usage", None)
        tool_calls: list[ToolCallResult] = []
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) != "function_call":
                continue
            tool_input: dict[str, Any] = {}
            arguments = getattr(item, "arguments", "")
            if arguments:
                try:
                    parsed = json.loads(arguments)
                    if isinstance(parsed, dict):
                        tool_input = parsed
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning(
                        "Malformed tool arguments for %s (id=%s): %s",
                        getattr(item, "name", ""),
                        getattr(item, "call_id", ""),
                        exc.__class__.__name__,
                    )
            tool_calls.append(
                ToolCallResult(
                    id=getattr(item, "call_id", ""),
                    name=getattr(item, "name", ""),
                    input=tool_input,
                )
            )

        cache_creation, cache_read = extract_openai_cache_tokens(usage)
        return CompletionResult(
            content=(
                content_override
                if content_override is not None
                else self._response_text(response)
            ),
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            finish_reason=self._finish_reason(response, bool(tool_calls)),
            tool_calls=tool_calls,
            thinking_content=extract_openai_reasoning_content(response),
            reasoning_details=extract_openai_reasoning_details(response),
            raw_response=response,
        )

    @staticmethod
    def _response_text(response: Any) -> str:
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text:
            return text
        refusal_parts: list[str] = []
        text_parts: list[str] = []
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) != "message":
                continue
            for part in getattr(item, "content", None) or []:
                part_type = getattr(part, "type", None)
                value = getattr(part, "text", None)
                if part_type == "output_text" and isinstance(value, str):
                    text_parts.append(value)
                refusal = getattr(part, "refusal", None)
                if part_type == "refusal" and isinstance(refusal, str):
                    refusal_parts.append(refusal)
        return "".join(text_parts) or "\n".join(refusal_parts)

    @staticmethod
    def _response_has_tool_calls(response: Any) -> bool:
        return any(
            getattr(item, "type", None) == "function_call"
            for item in (getattr(response, "output", None) or [])
        )

    @staticmethod
    def _finish_reason(response: Any, has_tool_calls: bool = False) -> str:
        if has_tool_calls:
            return "tool_calls"
        status = getattr(response, "status", None)
        if status == "incomplete":
            reason = getattr(
                getattr(response, "incomplete_details", None), "reason", None
            )
            if reason == "max_output_tokens":
                return "length"
            return reason or "incomplete"
        if status == "failed":
            return "error"
        return "stop"

    @staticmethod
    def _structured_output_mode(extra_params: dict[str, Any] | None) -> str | None:
        return extra_params.get("structured_output_mode") if extra_params else None

    @staticmethod
    def _with_json_schema_instructions(
        messages: list[dict[str, Any]], response_format: type[BaseModel]
    ) -> list[dict[str, Any]]:
        instruction = _json_object_instruction(response_format)
        new_messages = [dict(message) for message in messages]
        first = new_messages[0] if new_messages else None
        if (
            first
            and first.get("role") == "system"
            and isinstance(first.get("content"), str)
        ):
            first["content"] = f"{first['content']}\n\n{instruction}".strip()
        else:
            new_messages.insert(0, {"role": "system", "content": instruction})
        return new_messages

    @classmethod
    def _parse_or_repair_structured_content(
        cls,
        response: Any,
        response_format: type[BaseModel],
        model: str,
        *,
        empty_on_missing: bool,
    ) -> BaseModel | str:
        raw_content = cls._response_text(response)
        if raw_content:
            try:
                return validate_structured_output(raw_content, response_format)
            except (StructuredOutputError, ValidationError):
                return repair_response_model_json(raw_content, response_format, model)
        if not empty_on_missing:
            raise ValidationException("No parsed content in structured response")
        try:
            return empty_structured_output(response_format)
        except ValidationError:
            return ""

    @staticmethod
    def _structured_rejection_result(
        response_format: type[BaseModel], model: str
    ) -> CompletionResult:
        logger.warning(
            "Structured output via json_schema rejected by model %s; set "
            "structured_output_mode=json_object if needed.",
            model,
        )
        try:
            content: Any = empty_structured_output(response_format)
        except ValidationError:
            content = ""
        return CompletionResult(content=content)
