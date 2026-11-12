"""Certified bracket for the exact log-normal marginals (Section 5 of the paper).

For each basket strike of the three- and six-asset examples:

  HV, BV     Algorithm 1 on the discretised marginals (as in three_asset.py);
  HV_ex      price of that portfolio under the exact marginals;
  HV_ex*     the same after refining the strikes against the exact
             marginals: a lower bound for the original problem;
  BV_ex      basket value of the BRA coupling reassembled with the exact
             marginals: an upper bound for the original problem.

Usage:
    python -m subreplicating.examples.exact_bracket 3 [restarts]
    python -m subreplicating.examples.exact_bracket 6 [restarts]
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from ..exact import hedging_value_exact, polish_strikes_exact, reassembled_basket_value
from ..marginals import lognormal_marginals
from ..pricing import call_price
from ..rearrangement import basket_call_objective, bra, ra
from ..subreplication import analytical_cash, port_value, refine_portfolio, sub_portfolio
from ._common import run_best_coupling

TIME_PERIOD = 125 / 250
N = 50_000
MAXITER = 1_000_000
N_PARTITION = 100_000
CASES = {
    3: (np.array([35.5, 20.0, 25.0]) / 100, [100, 105, 110, 115, 120], 64),
    6: (np.array([91.5, 75.5, 82.5, 35.5, 20.0, 15.0]) / 100, [100, 110, 120, 130, 140], 20),
}


def main(n: int, restarts: int | None = None, out: str | None = None) -> None:
    vols, strikes, default_restarts = CASES[n]
    restarts = default_restarts if restarts is None else restarts
    weights = np.full(n, 100.0 / n)
    prices = lognormal_marginals(weights, vols, TIME_PERIOD, N)
    asset_max = prices.max(axis=0)
    rows = []
    print(f"  {n} assets, N = {N}, restarts per config = {restarts}")
    print(f"  {'K':>4} {'HV':>9} {'BV':>9} {'HV_ex':>9} {'HV_ex*':>9} {'BV_ex':>9} {'gap':>8} {'check':>8}")
    for kb in map(float, strikes):
        t0 = time.time()
        prices_r, _ = run_best_coupling(
            prices, kb, [bra, ra], restarts, MAXITER, N_PARTITION,
            objectives=[None, basket_call_objective(kb)],
        )
        sr = sub_portfolio(prices_r, kb, n_partition=N_PARTITION)
        m, _ = refine_portfolio(prices_r, sr, kb)
        hv = port_value(prices_r, m, kb).value
        bv = call_price(prices_r.sum(axis=1), kb)
        hv_ex = hedging_value_exact(m, weights, vols, TIME_PERIOD, kb)
        m_ex, hv_ex_star = polish_strikes_exact(
            m, weights, vols, TIME_PERIOD, kb, unpark_at=asset_max
        )
        bv_ex = reassembled_basket_value(prices_r, weights, vols, TIME_PERIOD, kb)
        # quadrature check: recompute with four times as many nodes
        bv_ex_fine = reassembled_basket_value(
            prices_r, weights, vols, TIME_PERIOD, kb, nodes=64, nodes_kink=1024
        )
        rows.append({
            "kb": kb, "hv": hv, "bv": bv, "hv_ex": hv_ex, "hv_ex_star": hv_ex_star,
            "bv_ex": bv_ex, "bv_ex_fine": bv_ex_fine,
            "strikes": [x.tolist() for x in m], "strikes_ex": [x.tolist() for x in m_ex],
            "cash_ex": analytical_cash(m_ex, kb),
            "seconds": time.time() - t0,
        })
        print(f"  {kb:4.0f} {hv:9.6f} {bv:9.6f} {hv_ex:9.6f} {hv_ex_star:9.6f} {bv_ex:9.6f} "
              f"{bv_ex - hv_ex_star:8.1e} {abs(bv_ex - bv_ex_fine):8.1e}   "
              f"{[np.round(x, 4).tolist() for x in m_ex]}  [{time.time() - t0:.0f}s]", flush=True)
        if out:
            with open(out, "w") as fh:
                json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main(
        int(sys.argv[1]),
        int(sys.argv[2]) if len(sys.argv) > 2 else None,
        sys.argv[3] if len(sys.argv) > 3 else None,
    )
