"""Case 2 of the paper (one dominant asset): reproduces the Case 2 table.

For each example the script builds the coupling mu_c (X_1, X_2 co-monotone,
X_3 antimonotone to X_1 + X_2), reads off the crossings of the basket
S_c(u) = F_1^{-1}(u) + F_2^{-1}(u) + F_3^{-1}(1-u) with the strike, prices the
vanilla portfolio min(x1,a1) + min(x2,a2) + (x3-c)^+ - a1 - a2 of
Proposition 3.2 at the first downcrossing, and compares it with the smallest
basket value found by RA/BRA over several restarts and both objectives.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from subreplicating import bra, call_price, port_value, ra
from subreplicating.rearrangement import basket_call_objective

N = 50_000
RESTARTS = 4
STALL = 300
MAXITER = 300_000

_u = np.linspace(0.5 / N, 1 - 0.5 / N, N)
_z = norm.ppf(_u)


def lognormal(mean: float, vol: float, T: float) -> np.ndarray:
    return mean * np.exp(-vol * vol * T / 2 + vol * np.sqrt(T) * _z)


def analyse(name: str, cols: list[np.ndarray], kb: float) -> None:
    n = len(cols)
    Xc = np.column_stack(cols[:-1] + [cols[-1][::-1]])   # mu_c
    S = Xc.sum(axis=1)
    d = np.diff(np.sign(S - kb))
    down, up = np.flatnonzero(d < 0), np.flatnonzero(d > 0)
    v_c = call_price(S, kb)

    best = np.inf
    for seed in range(RESTARTS):
        for f in (None, basket_call_objective(kb)):
            for algo in (bra, ra):
                kw = {} if f is None else {"f": f}
                r = algo(np.column_stack(cols), maxiter=MAXITER, stalliter=STALL, seed=seed, **kw)
                best = min(best, call_price(r.X.sum(axis=1), kb))

    print(f"{name}  (K = {kb:g}, N = {N})")
    print(f"  crossings of S_c: down at u = {np.round(_u[down], 4).tolist()}, up at u = {np.round(_u[up], 4).tolist()}")
    print(f"  V(mu_c) = {v_c:.6f}   best BV over {RESTARTS}x2x2 RA/BRA runs = {best:.6f}")
    if len(down) == 0:
        print("  no downcrossing: the portfolio of Proposition 3.2 is not defined\n")
        return
    m = down[0]
    a = [float(Xc[m, i]) for i in range(n - 1)]
    c = kb - sum(a)
    strikes = [np.array([ai, kb]) for ai in a] + [np.array([0.0, c])]
    hv = port_value(Xc, strikes, kb).value
    print(f"  u* = {_u[m]:.4f}, a = {np.round(a, 4).tolist()}, c = {c:.4f}")
    print(f"  bound (13) = {hv:.6f}   int_0^u* (S_c - K) du = {(S[: m + 1] - kb).sum() / N:.6f}")
    print(f"  HV - best BV = {hv - best:+.2e}  ->  {'certified optimal' if abs(hv - best) < 1e-6 else 'gap'}\n")


def main() -> None:
    w = 100.0 / 3
    analyse("U[0,30], U[0,30], lognormal(mean 40, vol 100%, T=1)", [30 * _u, 30 * _u, lognormal(40.0, 1.0, 1.0)], 70.0)
    analyse("lognormal, sigma = (10%, 10%, 120%), T = 0.5", [lognormal(w, .10, .5), lognormal(w, .10, .5), lognormal(w, 1.20, .5)], 110.0)
    analyse("lognormal, sigma = (20%, 20%, 60%), T = 0.5", [lognormal(w, .20, .5), lognormal(w, .20, .5), lognormal(w, .60, .5)], 105.0)
    analyse("lognormal, sigma = (35.5%, 20%, 25%), T = 0.5", [lognormal(w, .355, .5), lognormal(w, .20, .5), lognormal(w, .25, .5)], 100.0)


if __name__ == "__main__":
    main()
