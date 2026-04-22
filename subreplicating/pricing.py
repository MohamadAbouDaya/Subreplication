"""Vanilla option pricing from simulated price vectors."""

from __future__ import annotations

import numpy as np


def call_price(prices: np.ndarray, strike: float) -> float:
    """Expected payoff of a European call: E[max(S - K, 0)]."""
    return float(np.mean(np.maximum(prices - strike, 0.0)))


def put_price(prices: np.ndarray, strike: float) -> float:
    """Expected payoff of a European put: E[max(K - S, 0)]."""
    return float(np.mean(np.maximum(strike - prices, 0.0)))
