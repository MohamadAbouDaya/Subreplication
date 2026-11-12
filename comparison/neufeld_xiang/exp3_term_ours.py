"""Our construction on one convex term of the Experiment 3 cost of Neufeld-Xiang.

Cost |<s, x> - t| on their first n marginals, with s the first n entries of
their first direction vector, rescaled to unit length.  Since
|z - t| = 2 (z - t)^+ - (z - t), bounds on the basket call give bounds on the
cost.  Asset i is y_i = s_i x_i + 10 |s_i| >= 0.
Usage: PYTHONPATH=<repo> python exp3_term_ours.py n restarts offset1 offset2 ...
(t = E<s, x> + offset)
"""
import json
import sys
import time

import numpy as np

from subreplicating.exact import (AffineMarginal, TruncatedNormalMixtureMarginal,
                                  polish_strikes_exact, reassembled_basket_value)
from subreplicating.examples._common import run_best_coupling
from subreplicating.pricing import call_price
from subreplicating.rearrangement import basket_call_objective, bra, ra
from subreplicating.subreplication import port_value, refine_portfolio, sub_portfolio

N = 50_000


def main(n, restarts, offsets):
    d = json.load(open("exp3_marginals.json"))
    c = json.load(open("exp3_cost.json"))
    s = np.asarray(c["weights_plus"], float)[0, :n]
    s = s / np.linalg.norm(s)
    base = [TruncatedNormalMixtureMarginal(m["weights"], m["means"], m["stds"], d["lb"], d["ub"])
            for m in d["marginals"][:n]]
    margs = [AffineMarginal(b, si, 10 * abs(si)) for b, si in zip(base, s)]
    shift = float(10 * np.abs(s).sum())
    mean_sx = float(sum(si * b.mean for si, b in zip(s, base)))
    prices = np.column_stack([mg.discretise(N) for mg in margs])
    print(f"n = {n}, s = {np.round(s, 4).tolist()}, E<s,x> = {mean_sx:.6f}", flush=True)
    print(f"{'t':>8} {'LB':>10} {'UB':>10} {'gap':>8} {'check':>8}   (bounds on E|<s,x> - t|)")
    rows = []
    for off in offsets:
        t0 = time.time()
        t = mean_sx + off
        kb = t + shift
        prices_r, _ = run_best_coupling(prices, kb, [bra, ra], restarts, 1_000_000, 100_000,
                                        objectives=[None, basket_call_objective(kb)])
        sr = sub_portfolio(prices_r, kb, n_partition=100_000)
        m, _ = refine_portfolio(prices_r, sr, kb)
        hv = port_value(prices_r, m, kb).value
        bv = call_price(prices_r.sum(axis=1), kb)
        m_ex, hv_ex = polish_strikes_exact(m, margs, kb=kb, unpark_at=prices.max(axis=0))
        bv_ex = reassembled_basket_value(prices_r, margs, kb=kb)
        bv_fine = reassembled_basket_value(prices_r, margs, kb=kb, nodes=64, nodes_kink=1024)
        lb, ub = 2 * hv_ex - (mean_sx - t), 2 * bv_ex - (mean_sx - t)
        rows.append({"t": t, "offset": off, "call_hv": hv, "call_bv": bv, "call_hv_ex": hv_ex,
                     "call_bv_ex": bv_ex, "lb": lb, "ub": ub, "s": s.tolist(),
                     "strikes_y": [x.tolist() for x in m_ex], "seconds": time.time() - t0})
        print(f"{t:8.4f} {lb:10.6f} {ub:10.6f} {ub - lb:8.1e} {2 * abs(bv_ex - bv_fine):8.1e}   "
              f"call: HV {hv:.6f} BV {bv:.6f} HV_ex {hv_ex:.6f} BV_ex {bv_ex:.6f}  "
              f"strikes(y) {[np.round(x, 3).tolist() for x in m_ex]} [{time.time() - t0:.0f}s]",
              flush=True)
        json.dump(rows, open(f"exp3_term_ours_{n}.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), [float(x) for x in sys.argv[3:]])
