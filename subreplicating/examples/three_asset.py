"""Three-asset basket option subreplication example.

Equal-weighted portfolio of 3 assets with implied vols 35.5%, 20%, 25%.
Computes subreplicating portfolios for several strike levels and compares
against the basket call price and the comonotonic upper bound.

Usage:
    python -m subreplicating.examples.three_asset
"""

from __future__ import annotations

import numpy as np

from ..marginals import lognormal_marginals
from ..rearrangement import bra, ra
from ..subreplication import sub_portfolio
from ._common import print_report, run_best_coupling

# -- Parameters ---------------------------------------------------------------
TIME_PERIOD = 125 / 250          # half a year
N = 50_000                       # number of scenarios
WEIGHTS = np.array([100 / 3, 100 / 3, 100 / 3])
IMPLIED_VOLS = np.array([35.5, 20.0, 25.0]) / 100
BRA_RESTARTS = 250
BRA_MAXITER = 1_000_000

N_PARTITION = 100_000
STRIKES = [100, 105, 110, 115, 120]


def main():
    print("=" * 60)
    print("  THREE-ASSET BASKET OPTION SUBREPLICATION")
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
            algos=[bra, ra],
            restarts=BRA_RESTARTS,
            maxiter=BRA_MAXITER,
            n_partition=N_PARTITION,
        )
        sr = sub_portfolio(prices_r, kb, n_partition=N_PARTITION)
        print_report(prices_r, kb, sr, fval)


if __name__ == "__main__":
    main()
