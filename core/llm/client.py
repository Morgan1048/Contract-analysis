from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache

import tiktoken
from requests.exceptions import RequestException

from core.providers.config import ProviderConfig
from core.providers.openai_compat import OpenAICompatClient
from core.pricing import cost_usd
from core.obs.trace import span, set_span_attributes
from core.providers.errors import BackendError, RequestBudgetExceeded


class PromptBudgetExceeded(RequestBudgetExceeded):
    pass


@lru_cache(maxsize=16)
def _encoding(model):
    if "gpt-oss" in model:
        return tiktoken.get_encoding("o200k_harmony")
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def _system_with_schema(system, schema, example):
    schema_text = json.dumps(schema, sort_keys=True)
    example_text = json.dumps(example if example is not None else _synthesize_example(schema), sort_keys=True)
    return f"{system}\nRequired JSON object fields and types: {schema_text}\nExample: {example_text}"


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str
    usage_estimated: bool = False
    cost_usd: float | None = 0.0
    unpriced_model: bool = False


class LLMClientError(ValueError):
    """Raised when an LLM response cannot be decoded as the requested JSON."""


def _content_text(content) -> str:
    """Normalize an OpenAI-compatible message ``content`` field (string, list of
    parts, or None) into plain text."""
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def _parse_tool_arguments(raw) -> dict:
    """Tool-call arguments arrive as a JSON-encoded string on every OpenAI-
    compatible backend; a malformed one must not crash the agent loop."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


_EXAMPLE_PLACEHOLDERS = {"string": "...", "number": 0.5, "integer": 1, "boolean": True,
                         "array": [], "object": {}}


def _synthesize_example(schema: dict) -> dict:
    """Build a concrete example when the caller has none of its own.

    A prior version rendered the *type map itself* as the "Example", so the model
    saw literal values like ``"verdict": "string"`` and could reasonably copy them
    back verbatim. This substitutes a plausible concrete value per declared type
    instead, so the example is never identical to the schema.
    """
    return {
        key: _EXAMPLE_PLACEHOLDERS.get(type_name, f"<{type_name}>") if isinstance(type_name, str) else type_name
        for key, type_name in schema.items()
    }


def _json_object(text: str) -> dict:
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            raise LLMClientError("LLM response did not contain a JSON object")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMClientError("LLM response contained invalid JSON") from exc
    if not isinstance(value, dict):
        raise LLMClientError("LLM response JSON must be an object")
    return value


class LLMClient:
    def __init__(self, cfg: ProviderConfig) -> None:
        self.cfg = cfg
        self._client = OpenAICompatClient(cfg)

    def complete_json(self, system: str, user: str, schema: dict, example: dict | None = None) -> tuple[dict, LLMResponse]:
        self.validate_prompt(system, user, schema, example)
        with span("llm:generate", observation_type="generation", model=self.cfg.model,
                  system=self.cfg.provider):
            return self._complete_json(system, user, schema, example)

    def validate_prompt(self, system, user, schema, example=None):
        if self.cfg.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        rendered = _system_with_schema(system, schema, example)
        try:
            encoder = _encoding(self.cfg.model)
        except (RequestException, OSError) as exc:
            raise BackendError("Tokenizer unavailable; run setup.sh --install-only to populate the local tokenizer cache.") from exc
        count = len(encoder.encode(rendered, disallowed_special=())) + len(encoder.encode(user, disallowed_special=())) + 128
        if count > self.cfg.max_input_tokens:
            raise PromptBudgetExceeded(
                f"prompt_token_budget: estimated input {count} exceeds configured {self.cfg.max_input_tokens}; "
                "narrow evidence retrieval or configure a higher-capacity local/provider model."
            )
        return count

    def _complete_json(self, system, user, schema, example):
        system = _system_with_schema(system, schema, example)
        started = time.perf_counter()
        options = {"max_tokens": 2048}
        if "gpt-oss" in self.cfg.model.lower():
            options["reasoning_effort"] = "low"
        set_span_attributes(input={"system": system, "user": user},
                            **{"langfuse.observation.model.parameters": {"temperature": 0, **options}})
        text, usage = self._client.complete_with_usage(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            response_format={"type": "json_object"},
            **options,
        )
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        estimated = prompt_tokens is None or completion_tokens is None
        prompt_tokens = prompt_tokens if prompt_tokens is not None else max(1, len(system.split()) + len(user.split()))
        completion_tokens = completion_tokens if completion_tokens is not None else max(1, len(text.split()))
        cost = cost_usd(
            self.cfg.model,
            prompt_tokens,
            completion_tokens,
            is_local=self.cfg.is_local,
            provider=self.cfg.provider,
        )
        response = LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=max(0.001, (time.perf_counter() - started) * 1000),
            model=self.cfg.model,
            usage_estimated=estimated,
            cost_usd=cost,
            unpriced_model=cost is None,
        )
        set_span_attributes(output=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                            cost_usd=cost, usage_estimated=estimated, unpriced_model=cost is None)
        try:
            payload = _json_object(text)
        except LLMClientError as exc:
            exc.response = response
            raise
        return payload, response

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        with span("llm:chat", observation_type="generation", model=self.cfg.model,
                  system=self.cfg.provider, input={"messages": messages, "tools": tools}):
            return self._chat(messages, tools)

    def _chat(self, messages, tools):
        """OpenAI-compatible tool-calling chat step, used by ``ChatAgent``.

        Passes ``tools``/``tool_choice="auto"`` when tools are offered, and maps
        any ``message.tool_calls`` the backend returns into ``ToolCallRequest``.
        When the model responds with no tool calls, its content is the final
        answer: required to be a JSON object with ``answer`` and ``citations``
        (see ``ChatAgent``'s system prompt). A response that fails to parse as
        that shape is returned with ``parse_failed=True`` rather than silently
        treated as an answer with zero citations, so downstream quote
        verification cannot become a no-op.
        """
        # Imported here (not at module scope) to avoid a circular import: this
        # module is imported by core.pipeline, which core.agents.chat does not
        # depend on, but keeping the dependency one-directional and lazy is safer.
        from core.agents.chat import ChatResponse, ToolCallRequest

        extra: dict = {}
        if tools:
            extra["tools"] = tools
            extra["tool_choice"] = "auto"
        message, _usage = self._client.chat_completion_message(messages, **extra)
        prompt, completion = _usage.get("prompt_tokens"), _usage.get("completion_tokens")
        price = (cost_usd(self.cfg.model, prompt, completion, is_local=self.cfg.is_local, provider=self.cfg.provider)
                 if prompt is not None and completion is not None else None)
        set_span_attributes(output=message, prompt_tokens=prompt, completion_tokens=completion,
                            cost_usd=price, unpriced_model=price is None)

        raw_tool_calls = message.get("tool_calls") or []
        if raw_tool_calls:
            tool_calls = [
                ToolCallRequest(
                    id=call.get("id") or "",
                    name=(call.get("function") or {}).get("name") or "",
                    arguments=_parse_tool_arguments((call.get("function") or {}).get("arguments")),
                )
                for call in raw_tool_calls
            ]
            return ChatResponse(content=_content_text(message.get("content")) or None, tool_calls=tool_calls)

        content_text = _content_text(message.get("content"))
        try:
            parsed = _json_object(content_text)
            answer = parsed.get("answer")
            citations = parsed.get("citations")
            if not isinstance(answer, str) or not isinstance(citations, list):
                raise LLMClientError("final answer JSON missing 'answer'/'citations' fields")
        except LLMClientError:
            return ChatResponse(content=content_text, citations=None, parse_failed=True)
        return ChatResponse(content=answer, citations=citations, parse_failed=False)
