from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RearrangementResult:
    """Result returned by ra(), bra(), and brave()."""

    X: np.ndarray               # rearranged matrix (rows sorted by decreasing row-sum)
    S: np.ndarray               # row sums (descending)
    niter: int                  # total iterations performed
    fval: float                 # final objective value
    fiter: np.ndarray           # objective value per iteration
    converged: bool             # whether convergence criterion was met
    rand: np.ndarray | None = None          # (brave only) which iterations used a random partition
    blocksizeiter: np.ndarray | None = None # (brave only) block size per iteration


@dataclass
class SubportfolioResult:
    """Result returned by sub_portfolio()."""

    cash: float                 # cash component (from Monte Carlo)
    v: float                    # vanilla-options component
    value: float                # total = cash + v
    m: list[np.ndarray]         # strikes per asset
    phi: list[np.ndarray]       # phi function per asset


@dataclass
class PortfolioValueResult:
    """Result returned by port_value()."""

    cash: float
    v: float
    value: float
