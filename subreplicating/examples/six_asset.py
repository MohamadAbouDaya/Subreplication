"""Six-asset basket option subreplication example (Algorithm 1 of the paper).

Equal-weighted portfolio of 6 assets, S_0 = 100 each, implied vols
91.5%, 75.5%, 82.5%, 35.5%, 20%, 15%, maturity half a year.  Same pipeline as
``three_asset.py`` (RA and BRA, variance and K-aware objectives, Nelder-Mead
polish), which is what Table 2 of the paper reports.

Usage:
    python -m subreplicating.examples.six_asset
"""

from __future__ import annotations

import numpy as np

from ..marginals import lognormal_marginals
from ..rearrangement import basket_call_objective, bra, ra
from ._common import print_report, run_pipeline

# -- Parameters ---------------------------------------------------------------
TIME_PERIOD = 125 / 250
N = 50_000
WEIGHTS = np.full(6, 100.0 / 6)
IMPLIED_VOLS = np.array([91.5, 75.5, 82.5, 35.5, 20.0, 15.0]) / 100
RESTARTS = 20                    # per (algorithm, objective) pair
MAXITER = 1_000_000

N_PARTITION = 100_000
STRIKES = [100, 110, 120, 130, 140]


def main():
    print("=" * 64)
    print("  SIX-ASSET BASKET OPTION SUBREPLICATION (Algorithm 1)")
    print("=" * 64)
    print(f"  N = {N}, T = {TIME_PERIOD}, restarts per config = {RESTARTS}")
    print(f"  Weights: {np.round(WEIGHTS, 2)}")
    print(f"  Implied vols: {IMPLIED_VOLS}")
    print("  Search: [BRA, RA] x [variance, K-aware] + Nelder-Mead polish")
    print()

    prices = lognormal_marginals(WEIGHTS, IMPLIED_VOLS, TIME_PERIOD, N)

    for kb in STRIKES:
        kb_f = float(kb)
        res = run_pipeline(
            prices, kb_f,
            algos=[bra, ra],
            objectives=[None, basket_call_objective(kb_f)],
            restarts=RESTARTS,
            maxiter=MAXITER,
            n_partition=N_PARTITION,
        )
        print_report(res)


if __name__ == "__main__":
    main()
