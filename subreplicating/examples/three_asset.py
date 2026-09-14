"""Three-asset basket option subreplication example (Algorithm 1 of the paper).

Equal-weighted portfolio of 3 assets, S_0 = 100 each, implied vols 35.5%, 20%,
25%, maturity half a year.  For each basket strike: RA and BRA are run with the
variance objective and with the K-aware objective E[(sum - K)^+], the coupling
with the highest raw hedging value is kept, the phi-crossing strikes are
polished with Nelder-Mead, and HV is compared with the basket value BV under
the selected coupling (Corollary 2.4 of the paper: HV = BV certifies
optimality for the discretised marginals).

Usage:
    python -m subreplicating.examples.three_asset
"""

from __future__ import annotations

import numpy as np

from ..marginals import lognormal_marginals
from ..rearrangement import basket_call_objective, bra, ra
from ._common import print_report, run_pipeline

# -- Parameters ---------------------------------------------------------------
TIME_PERIOD = 125 / 250          # half a year
N = 50_000                       # number of atoms per marginal
WEIGHTS = np.array([100 / 3, 100 / 3, 100 / 3])   # w_i * S_{i,0}
IMPLIED_VOLS = np.array([35.5, 20.0, 25.0]) / 100
RESTARTS = 250                   # per (algorithm, objective) pair
MAXITER = 1_000_000

N_PARTITION = 100_000
STRIKES = [100, 105, 110, 115, 120]


def main():
    print("=" * 64)
    print("  THREE-ASSET BASKET OPTION SUBREPLICATION (Algorithm 1)")
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
