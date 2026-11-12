"""Independent check of the Neufeld-Xiang output (does not use MATLAB).

For every run in <tag>.json:
  * re-price the dual functions h_i under the truncated log-normal marginals;
  * recompute  min_x [(sum x - K)^+ - sum_i h_i(x_i) - const]  over the box,
    by exact vertex enumeration (n = 3) and by a HiGHS mixed-integer programme
    (any n);  a value >= 0 (up to solver tolerance) means the dual is feasible;
  * print the resulting verified lower bound next to the reported LB / UB.
"""
import itertools
import json
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from scipy.stats import norm


def cdf_diff(a, b):
    out = norm.cdf(b) - norm.cdf(a)
    up = a > 0
    out[up] = norm.cdf(-a[up]) - norm.cdf(-b[up])
    return out


def price_pwl(knots, vals, mu, s, norm_const):
    """E[h(X)] for h piecewise linear through (knots, vals), X truncated log-normal."""
    k1, k2 = knots[:-1], knots[1:]
    a = np.diff(vals) / np.diff(knots)
    b = vals[:-1] - a * k1
    with np.errstate(divide="ignore"):
        d1 = (np.log(k1) - mu) / s
        d2 = (np.log(k2) - mu) / s
    m = np.exp(mu + 0.5 * s * s)
    return float(np.sum(a * m * cdf_diff(d1 - s, d2 - s) + b * cdf_diff(d1, d2)) / norm_const)


def min_enum3(knots, vals, K):
    """Exact minimum of (sum x - K)^+ - sum h_i(x_i) over the box, n = 3."""
    best = np.inf
    k0, k1, k2 = knots
    v0, v1, v2 = vals
    s01 = k0[:, None] + k1[None, :]
    h01 = v0[:, None] + v1[None, :]
    for j in range(len(k2)):                       # grid vertices
        val = np.maximum(s01 + k2[j] - K, 0.0) - h01 - v2[j]
        best = min(best, val.min())
    for free in range(3):                          # vertices on {sum x = K}
        others = [i for i in range(3) if i != free]
        ka, kb = knots[others[0]], knots[others[1]]
        xf = K - ka[:, None] - kb[None, :]
        ok = (xf >= knots[free][0]) & (xf <= knots[free][-1])
        hf = np.interp(xf, knots[free], vals[free])
        val = -(vals[others[0]][:, None] + vals[others[1]][None, :] + hf)
        if ok.any():
            best = min(best, val[ok].min())
    return float(best)


def min_milp(knots, vals, K):
    n = len(knots)
    sizes = [len(k) for k in knots]
    off_l = np.concatenate([[0], np.cumsum(sizes)])            # lambda blocks
    nl = off_l[-1]
    off_z = nl + np.concatenate([[0], np.cumsum([m - 1 for m in sizes])])
    nz = off_z[-1] - nl
    t = nl + nz
    nv = t + 1
    c = np.zeros(nv)
    c[t] = 1.0
    rows, lo, hi = [], [], []
    A = lil_matrix((2 * n + sum(sizes) + 1, nv))
    r = 0
    for i in range(n):
        m = sizes[i]
        c[off_l[i]:off_l[i] + m] = -vals[i]
        A[r, off_l[i]:off_l[i] + m] = 1.0; lo.append(1.0); hi.append(1.0); r += 1
        A[r, off_z[i]:off_z[i] + m - 1] = 1.0; lo.append(1.0); hi.append(1.0); r += 1
        for k in range(m):                                      # SOS2 link
            A[r, off_l[i] + k] = 1.0
            if k > 0:
                A[r, off_z[i] + k - 1] = -1.0
            if k < m - 1:
                A[r, off_z[i] + k] = -1.0
            lo.append(-np.inf); hi.append(0.0); r += 1
    for i in range(n):                                          # t >= sum x - K
        A[r, off_l[i]:off_l[i] + sizes[i]] = knots[i]
    A[r, t] = -1.0; lo.append(-np.inf); hi.append(K); r += 1
    integrality = np.zeros(nv)
    integrality[nl:nl + nz] = 1
    res = milp(c, constraints=LinearConstraint(A.tocsr()[:r], lo, hi),
               integrality=integrality,
               bounds=Bounds(np.zeros(nv), np.concatenate([np.ones(nl + nz), [np.inf]])),
               options={"mip_rel_gap": 0.0, "time_limit": 600})
    bound = getattr(res, "mip_dual_bound", None)
    return (float(res.fun) if res.fun is not None else np.nan,
            float(bound) if bound is not None else np.nan, res.status)


def main(tag):
    d = json.load(open(f"{tag}.json"))
    sig = np.atleast_1d(d["sig"]).astype(float)
    n = len(sig)
    s = sig * np.sqrt(d["T"])
    mu = np.log(d["scale"]) - 0.5 * s * s
    norm_const = 1 - 1e-12
    runs = d["runs"] if isinstance(d["runs"], list) else [d["runs"]]
    print(f"{'K':>5} {'m':>4} {'LB(NX)':>10} {'price':>10} {'min enum':>10} "
          f"{'min milp':>10} {'milp bnd':>10} {'LB verified':>11} {'UB(MC)':>9} {'se':>8} {'atoms':>5} {'iter':>5} {'time':>7}")
    for run in runs:
        knots = [np.asarray(k, float) for k in run["knots"]]
        vals = [np.asarray(v, float) for v in run["coefs"]]
        K = float(run["K"])
        const = float(run["constant"])
        with np.errstate(divide="ignore"):
            nc = [norm.cdf((np.log(knots[i][-1]) - mu[i]) / s[i]) - norm.cdf((np.log(knots[i][0]) - mu[i]) / s[i])
                  for i in range(n)]
        price = const + sum(price_pwl(knots[i], vals[i], mu[i], s[i], nc[i]) for i in range(n))
        me = min_enum3(knots, vals, K) - const if n == 3 else np.nan
        mm, mb, _ = min_milp(knots, vals, K)
        mm -= const; mb -= const
        worst = np.nanmin([me, mb])
        print(f"{K:5.0f} {run['m']:4d} {run['LB']:10.6f} {price:10.6f} {me:10.2e} "
              f"{mm:10.2e} {mb:10.2e} {price + min(worst, 0.0):11.6f} {run['UB_mean']:9.6f} "
              f"{run['UB_se']:8.1e} {len(np.atleast_1d(run['probs'])):5d} {run['iter']:5d} {run['time']:7.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
