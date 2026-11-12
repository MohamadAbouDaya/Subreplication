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

    cash: float                 # cash component (exact, vertex enumeration)
    v: float                    # vanilla-options component
    value: float                # total = cash + v
    m: list[np.ndarray]         # strikes per asset
    phi: list[np.ndarray]       # phi function per asset
    # Strikes found by the strike search (phi-crossings and parity choice) and
    # their hedging value, before the comparison with the strike-free
    # portfolios.  None when no strike search was carried out.  These are the
    # strikes to refine: their unrefined value can be below a strike-free
    # portfolio that the refined strikes beat (see ``refine_portfolio``).
    m_search: list[np.ndarray] | None = None
    value_search: float | None = None


@dataclass
class PortfolioValueResult:
    """Result returned by port_value()."""

    cash: float
    v: float
    value: float
