"""Our construction on the marginals of Experiment 3 of Neufeld-Xiang.

First n truncated normal mixtures of their experiment (support [-10, 10]),
cost (x_1 + ... + x_n - K)^+.  Internally each asset is shifted by +10 so that
it is non-negative; strikes are reported in the original scale.
Usage: PYTHONPATH=<repo> python exp3_ours.py n restarts K1 K2 ...
"""
import json
import sys
import time

import numpy as np

from subreplicating.exact import (TruncatedNormalMixtureMarginal, hedging_value_exact,
                                  polish_strikes_exact, reassembled_basket_value)
from subreplicating.examples._common import run_best_coupling
from subreplicating.pricing import call_price
from subreplicating.rearrangement import basket_call_objective, bra, ra
from subreplicating.subreplication import analytical_cash, port_value, refine_portfolio, sub_portfolio

N = 50_000


def main(n, restarts, strikes):
    d = json.load(open("exp3_marginals.json"))
    lb, ub = d["lb"], d["ub"]
    margs = [TruncatedNormalMixtureMarginal(m["weights"], m["means"], m["stds"], lb, ub, shift=lb)
             for m in d["marginals"][:n]]
    prices = np.column_stack([mg.discretise(N) for mg in margs])
    mean_sum = sum(mg.mean for mg in margs) + n * lb
    print(f"n = {n}, mean of the sum = {mean_sum:.4f}, comonotone sum range "
          f"[{n * lb:.0f}, {n * ub:.0f}]", flush=True)
    rows = []
    print(f"{'K':>6} {'HV':>9} {'BV':>9} {'HV_ex*':>9} {'BV_ex':>9} {'gap':>8} {'check':>8}")
    for K in strikes:
        t0 = time.time()
        kb = K - n * lb                                    # strike in the shifted scale
        prices_r, _ = run_best_coupling(prices, kb, [bra, ra], restarts, 1_000_000, 100_000,
                                        objectives=[None, basket_call_objective(kb)])
        sr = sub_portfolio(prices_r, kb, n_partition=100_000)
        m, _ = refine_portfolio(prices_r, sr, kb)
        hv = port_value(prices_r, m, kb).value
        bv = call_price(prices_r.sum(axis=1), kb)
        m_ex, hv_ex = polish_strikes_exact(m, margs, kb=kb, unpark_at=prices.max(axis=0))
        bv_ex = reassembled_basket_value(prices_r, margs, kb=kb)
        bv_fine = reassembled_basket_value(prices_r, margs, kb=kb, nodes=64, nodes_kink=1024)
        rows.append({"K": K, "kb": kb, "hv": hv, "bv": bv, "hv_ex": hv_ex, "bv_ex": bv_ex,
                     "strikes_shifted": [x.tolist() for x in m_ex],
                     "strikes": [(x + lb).tolist() for x in m_ex],
                     "cash": analytical_cash(m_ex, kb), "seconds": time.time() - t0})
        print(f"{K:6.2f} {hv:9.6f} {bv:9.6f} {hv_ex:9.6f} {bv_ex:9.6f} {bv_ex - hv_ex:8.1e} "
              f"{abs(bv_ex - bv_fine):8.1e}  {[np.round(x + lb, 3).tolist() for x in m_ex]} "
              f"[{time.time() - t0:.0f}s]", flush=True)
        json.dump(rows, open(f"exp3_ours_{n}.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), [float(x) for x in sys.argv[3:]])
