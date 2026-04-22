"""Tightest three-asset fit.

Same setup as `three_asset.py` but with three cumulative improvements:

  1. Both BRA and RA at every restart.
  2. Each algorithm runs with two objectives — the default variance proxy
     AND a K-aware objective E[(Σs - K)^+]. Because (Σs - K)^+ is convex, each
     counter-monotonic step remains non-increasing (supermodular rearrangement
     inequality), so the algorithms are valid for this objective. Best-by-HV
     across all runs wins.
  3. After strike selection, a Nelder-Mead polish maximises HV(m) directly over
     the strike values. HV is continuous and piecewise-linear in m, so the
     polish closes (a) the finite-grid phi-crossing discretisation error, and
     (b) the proxy misalignment between variance and HV.

Usage:
    python -m subreplicating.examples.three_asset_best
"""

from __future__ import annotations

import numpy as np

from ..marginals import lognormal_marginals
from ..pricing import call_price
from ..rearrangement import basket_call_objective, bra, ra
from ..subreplication import polish_strikes, port_value, sub_portfolio
from ._common import run_best_coupling

# -- Parameters ---------------------------------------------------------------
TIME_PERIOD = 125 / 250
N = 50_000
WEIGHTS = np.array([100 / 3, 100 / 3, 100 / 3])
IMPLIED_VOLS = np.array([35.5, 20.0, 25.0]) / 100
RESTARTS = 250
MAXITER = 1_000_000

N_PARTITION = 100_000
STRIKES = [100, 105, 110, 115, 120]


def main():
    print("=" * 64)
    print("  THREE-ASSET BASKET OPTION — BEST-FIT SEARCH")
    print("=" * 64)
    print(f"  N = {N}, T = {TIME_PERIOD}, restarts per config = {RESTARTS}")
    print(f"  Weights: {np.round(WEIGHTS, 2)}")
    print(f"  Implied vols: {IMPLIED_VOLS}")
    print("  Search configs: [BRA, RA] × [variance, K-aware] + Nelder-Mead polish")
    print()

    prices = lognormal_marginals(WEIGHTS, IMPLIED_VOLS, TIME_PERIOD, N)

    print(f"  {'K':>4} | {'BV':>9} {'HV_raw':>9} {'HV_polish':>10} {'gap':>7}")
    print("  " + "-" * 48)
    for kb in STRIKES:
        kb_f = float(kb)
        prices_r, _ = run_best_coupling(
            prices, kb_f,
            algos=[bra, ra],
            objectives=[None, basket_call_objective(kb_f)],
            restarts=RESTARTS,
            maxiter=MAXITER,
            n_partition=N_PARTITION,
        )
        sr = sub_portfolio(prices_r, kb_f, n_partition=N_PARTITION)
        m_pol, hv_pol = polish_strikes(prices_r, sr.m, kb_f)
        pv = port_value(prices_r, m_pol, kb_f)
        bv = call_price(prices_r.sum(axis=1), kb_f)
        gap = (bv - hv_pol) / bv * 100 if bv > 1e-8 else 0.0

        print(f"  {kb:>4} | {bv:>9.5f} {sr.value:>9.5f} {hv_pol:>10.6f} {gap:>6.2f}%")
        print(f"       strikes: {[np.round(x, 4).tolist() for x in m_pol]}")
        print(f"       cash = {pv.cash:.4f}, v = {pv.v:.4f}")
        print()


if __name__ == "__main__":
    main()
