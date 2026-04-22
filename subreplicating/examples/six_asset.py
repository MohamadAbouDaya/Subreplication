"""Six-asset basket option subreplication example.

Equal-weighted portfolio of 6 assets with implied vols
91.5%, 75.5%, 82.5%, 35.5%, 20%, 15%.
Computes the subreplicating portfolio for several strike levels.

Usage:
    python -m subreplicating.examples.six_asset
"""

from __future__ import annotations

import numpy as np

from ..marginals import lognormal_marginals
from ..rearrangement import bra
from ..subreplication import sub_portfolio
from ._common import print_report, run_best_coupling

# -- Parameters ---------------------------------------------------------------
TIME_PERIOD = 125 / 250
N = 50_000
WEIGHTS = np.full(6, 100.0 / 6)
IMPLIED_VOLS = np.array([91.5, 75.5, 82.5, 35.5, 20.0, 15.0]) / 100
BRA_RESTARTS = 20
BRA_MAXITER = 1_000_000

N_PARTITION = 100_000
STRIKES = [100, 110, 120, 130, 140]


def main():
    print("=" * 60)
    print("  SIX-ASSET BASKET OPTION SUBREPLICATION")
    print("=" * 60)
    print(f"  N = {N}, T = {TIME_PERIOD}, BRA restarts = {BRA_RESTARTS}")
    print(f"  Weights: {np.round(WEIGHTS, 2)}")
    print(f"  Implied vols: {IMPLIED_VOLS}")
    print()

    prices = lognormal_marginals(WEIGHTS, IMPLIED_VOLS, TIME_PERIOD, N)

    for kb in STRIKES:
        print("-" * 60)
        print(f"  Basket strike K = {kb}")
        print("-" * 60)

        prices_r, fval = run_best_coupling(
            prices, kb,
            algos=[bra],
            restarts=BRA_RESTARTS,
            maxiter=BRA_MAXITER,
            n_partition=N_PARTITION,
        )
        sr = sub_portfolio(prices_r, kb, n_partition=N_PARTITION)
        # Note: sub_portfolio already extends the last strike to kb when the
        # asset cannot reach it, so no post-processing is needed here.
        print_report(prices_r, kb, sr, fval)


if __name__ == "__main__":
    main()
