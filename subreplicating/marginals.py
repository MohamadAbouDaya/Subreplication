"""Utilities for generating marginal price distributions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


def lognormal_marginals(
    weights: NDArray,
    vols: NDArray,
    T: float,
    n: int,
) -> NDArray:
    """Generate lognormal marginal price samples at midpoint quantiles.

    Returns an (n, d) array where column *i* holds `n` equally-spaced
    quantiles of a lognormal marginal with volatility `vols[i]`, scaled by
    `weights[i]`. Midpoint quantiles avoid the 0/1 boundary where
    norm.ppf diverges.
    """
    u = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    z = norm.ppf(u)[:, None]
    vols_row = vols[None, :]
    drift = -vols_row**2 * T / 2
    return weights[None, :] * np.exp(drift + vols_row * np.sqrt(T) * z)
