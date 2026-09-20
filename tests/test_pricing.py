import pytest

from core.pricing import cost_usd


@pytest.mark.parametrize(
    ("dated_model", "base_model", "expected"),
    [
        ("gpt-4o-mini-2024-07-18", "gpt-4o-mini", 0.75),
        ("gpt-4o-2024-08-06", "gpt-4o", 20.0),
    ],
)
def test_dated_model_prices_match_base_ids(dated_model, base_model, expected):
    base_cost = cost_usd(base_model, 1_000_000, 1_000_000, provider="openai")
    assert base_cost == expected
    assert cost_usd(dated_model, 1_000_000, 1_000_000, provider="openai") == base_cost
