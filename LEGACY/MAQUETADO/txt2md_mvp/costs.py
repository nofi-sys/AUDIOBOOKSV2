from __future__ import annotations

from typing import Dict, Optional

PRICING_DATA: Dict[str, Dict[str, float]] = {
    "gpt-5": {
        "input": 1.250 / 1_000_000,
        "cached_input": 0.125 / 1_000_000,
        "output": 10.000 / 1_000_000,
    },
    "gpt-5-mini": {
        "input": 0.250 / 1_000_000,
        "cached_input": 0.025 / 1_000_000,
        "output": 2.000 / 1_000_000,
    },
    "gpt-5-nano": {
        "input": 0.050 / 1_000_000,
        "cached_input": 0.005 / 1_000_000,
        "output": 0.400 / 1_000_000,
    },
    "gpt-5-pro": {
        "input": 15.00 / 1_000_000,
        "cached_input": 0,  # No cache price provided
        "output": 120.00 / 1_000_000,
    },
}

def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    pricing_data: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """Calculates the estimated cost for an AI operation."""
    pricing_table = pricing_data or PRICING_DATA
    if model not in pricing_table:
        return 0.0

    prices = pricing_table[model]

    input_cost = input_tokens * prices["input"]
    output_cost = output_tokens * prices["output"]
    cached_cost = cached_input_tokens * prices.get("cached_input", 0)

    return input_cost + output_cost + cached_cost
