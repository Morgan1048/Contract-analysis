"""Provider pricing. Local inference has no dollar charge; electricity and latency remain costs."""

from __future__ import annotations


ZERO_RATED_USD_PER_MILLION = {
    # Groq free-tier calls are genuinely zero-dollar API usage within the
    # account's rate limits. Keeping these in a separate table makes the
    # distinction from "unknown/unpriced" visible to callers and reviewers.
    "groq:openai/gpt-oss-120b": (0.0, 0.0),
    "groq:gpt-oss-120b": (0.0, 0.0),
}

PRICES_USD_PER_MILLION = {
    "gpt-4o": (5.0, 15.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.0, 8.0),
    # Embedding models bill only on input tokens; the completion-side rate is 0.
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "text-embedding-ada-002": (0.10, 0.0),
    # Provider-qualified entries avoid mispricing the same model id on another
    # OpenAI-compatible host with different billing terms.
    "siliconflow:openai/gpt-oss-120b": (0.05, 0.45),
    "siliconflow:gpt-oss-120b": (0.05, 0.45),
    "siliconflow:qwen/qwen3.5-9b": (0.10, 0.15),
    "siliconflow:deepseek-ai/deepseek-v4-flash": (0.13, 0.28),
    "siliconflow:baai/bge-m3": (0.02, 0.0),
    **ZERO_RATED_USD_PER_MILLION,
}


def _price_candidates(model: str, provider: str | None) -> list[str]:
    model_key = model.strip().lower()
    if provider is None:
        return [model_key]
    provider_key = provider.strip().lower()
    if provider_key == "unknown":
        return []
    return [f"{provider_key}:{model_key}", model_key]


def cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    is_local: bool = False,
    provider: str | None = None,
    backend: str | None = None,
) -> float | None:
    """Return the dollar cost for a completion, or ``None`` when it cannot be priced.

    A local backend costs zero dollars by definition. An unrecognised *non-local*
    model is a data-quality problem, not a free one: reporting it as ``0.0`` would be
    indistinguishable from an actual local run and would silently hide real spend.
    Callers must treat ``None`` as "unpriced", not as zero.
    """
    if is_local:
        return 0.0
    candidates = _price_candidates(model, provider if provider is not None else backend)
    if not candidates:
        return None
    rates = next((PRICES_USD_PER_MILLION[name] for name in candidates if name in PRICES_USD_PER_MILLION), None)
    if rates is None:
        rates = next(
            (
                rates
                for name, rates in sorted(PRICES_USD_PER_MILLION.items(), key=lambda item: len(item[0]), reverse=True)
                if ":" not in name and name in candidates[-1]
            ),
            None,
        )
    if rates is None:
        return None
    return round(prompt_tokens * rates[0] / 1_000_000 + completion_tokens * rates[1] / 1_000_000, 8)
