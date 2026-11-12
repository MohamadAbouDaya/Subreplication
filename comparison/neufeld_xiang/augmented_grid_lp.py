"""Best sub-replicating portfolio with strikes on an augmented grid (Section 5.2).

On a grid of knots the lower bound of Neufeld and Xiang is, up to the tolerance
of their semi-infinite programme, the value of

    sup  sum_i E[h_i(X_i)]   subject to   sum_i h_i(x_i) <= (sum_i x_i - K)^+

on the box prod_i [0, ub_i], over continuous functions h_i that are piecewise
linear with kinks at the knots of asset i (the marginals are the log-normal
laws truncated at ub_i, their 1 - 1e-12 quantile).  The left-hand side minus the
right-hand side is piecewise affine on the cells cut out by the knots and by the
hyperplane {sum x = K}, so the constraint holds on the box if and only if it
holds at the vertices of these cells:

  * the grid points (every coordinate at a knot), and
  * the points of {sum x = K} with all coordinates but one at a knot.

The programme is therefore a finite linear programme.  It is solved here by
constraint generation: HiGHS solves the LP on a subset of the vertices, an exact
enumeration of all vertices returns the most violated ones, and these are added
until no vertex is violated.  The value V of the last LP is an upper estimate of
the optimum (it uses a subset of the constraints) and V + d, with d <= 0 the
smallest slack over all vertices, is the value of a feasible portfolio; the
optimum lies between the two.

The grid is the one built by ``run_basket_knots.m``: m equal-probability knots
per asset, augmented with the strikes of our portfolio.

Usage (from this folder, with the package importable):
    python augmented_grid_lp.py 3 [m ...]    # strikes of results/final3.json; default m = 4 16
    python augmented_grid_lp.py 6 [m ...]
    python augmented_grid_lp.py --check fk3  # on the knots stored in results/fk3.json:
                                             # compare with the bound returned by their code
"""
import itertools
import json
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.special import ndtri

from verify import price_pwl

HERE = pathlib.Path(__file__).resolve().parent
T = 0.5
VOLS = {3: [0.355, 0.20, 0.25], 6: [0.915, 0.755, 0.825, 0.355, 0.20, 0.15]}
TRUNC = 1e-12          # the marginals are truncated at their 1 - TRUNC quantile
BIG = 1e4              # box on the LP variables; inactive at the optimum (checked)


def marginal_parameters(n):
    s = np.asarray(VOLS[n]) * np.sqrt(T)
    mu = np.log(100.0 / n) - 0.5 * s * s
    ub = np.exp(mu + s * ndtri(1 - TRUNC))
    return mu, s, ub


def build_knots(mu, s, ub, m, extra):
    """Knots of run_basket_knots.m: m equal-probability knots plus the extra ones."""
    out = []
    for i in range(len(mu)):
        p = np.linspace(0.0, 1.0, m + 1)
        with np.errstate(divide="ignore"):
            k = np.exp(mu[i] + s[i] * ndtri(p * (1 - TRUNC)))
        k[0], k[-1] = 0.0, ub[i]
        add = np.asarray(extra[i], float)
        add = add[(add > 1e-6) & (add < ub[i] - 1e-6)]
        k = np.unique(np.concatenate([k, add]))
        k = k[np.concatenate([[True], np.diff(k) > 1e-6])]
        k[-1] = ub[i]
        out.append(k)
    return out


def hat_prices(knots, mu, s):
    """E[g_ij(X_i)] for the hat function g_ij of knot j of asset i (truncated law)."""
    w = []
    for i, k in enumerate(knots):
        wi = np.empty(len(k))
        for j in range(len(k)):
            e = np.zeros(len(k))
            e[j] = 1.0
            wi[j] = price_pwl(k, e, mu[i], s[i], 1 - TRUNC)
        w.append(wi)
    return w


class Vertices:
    """Exact enumeration of the vertices, split into two blocks of assets."""

    def __init__(self, knots, K):
        self.knots, self.K, self.n = knots, K, len(knots)
        self.off = np.concatenate([[0], np.cumsum([len(k) for k in knots])])

    def _block(self, assets):
        """All knot combinations of a block: (index array, sum of the coordinates)."""
        if not assets:
            return np.zeros((1, 0), dtype=np.int64), np.zeros(1)
        idx = np.array(list(itertools.product(*[range(len(self.knots[i])) for i in assets])), dtype=np.int64)
        sums = sum(self.knots[a][idx[:, c]] for c, a in enumerate(assets))
        return idx, sums

    def _values(self, idx, assets, v):
        return sum(v[self.off[a] + idx[:, c]] for c, a in enumerate(assets)) if assets else np.zeros(len(idx))

    def axis_items(self):
        """Grid points with one coordinate at a knot and the others at their first knot
        (one per LP variable, so that every variable appears in a constraint)."""
        A = list(range(self.n))
        items = []
        for i in range(self.n):
            for j in range(len(self.knots[i])):
                ia = np.zeros(self.n, dtype=np.int64)
                ia[i] = j
                items.append((0.0, None, A, [], ia, np.zeros(0, dtype=np.int64), None))
        return items

    def worst(self, v, keep, chunk=400, only_violated=True):
        """Slack rhs - lhs at every vertex; returns (smallest slack, the `keep` vertices of
        smallest slack, restricted to the violated ones when ``only_violated``)."""
        found = []                                   # (slack, kind, free asset, idx tuple, x_free)
        smallest = np.inf
        for free in [None, *range(self.n)]:
            assets = [i for i in range(self.n) if i != free]
            half = (len(assets) + 1) // 2
            A, B = assets[:half], assets[half:]
            iA, sA = self._block(A)
            iB, sB = self._block(B)
            hA, hB = self._values(iA, A, v), self._values(iB, B, v)
            for start in range(0, len(iA), chunk):
                sl = slice(start, start + chunk)
                tot = sA[sl, None] + sB[None, :]
                if free is None:
                    slack = np.maximum(tot - self.K, 0.0) - hA[sl, None] - hB[None, :]
                    ok = np.ones(slack.shape, bool)
                    xf = None
                else:
                    kf = self.knots[free]
                    xf = self.K - tot
                    ok = (xf >= kf[0]) & (xf <= kf[-1])
                    hf = np.interp(xf, kf, v[self.off[free]:self.off[free + 1]])
                    slack = np.where(ok, -hA[sl, None] - hB[None, :] - hf, np.inf)
                if not ok.any():
                    continue
                smallest = min(smallest, float(slack.min()))
                flat = slack.ravel()
                k = min(max(50, keep // 20), flat.size)      # per chunk; the global cut is below
                pick = np.argpartition(flat, k - 1)[:k]
                pick = pick[flat[pick] < (-1e-11 if only_violated else np.inf)]
                for q in pick:
                    a, b = divmod(int(q), slack.shape[1])
                    found.append((float(flat[q]), free, A, B, iA[start + a], iB[b],
                                  None if xf is None else float(xf[a, b])))
        found.sort(key=lambda t: t[0])
        return smallest, found[:keep]

    def row(self, item):
        """Sparse constraint row (columns, coefficients, right-hand side) of a vertex."""
        _, free, A, B, ia, ib, xf = item
        cols = [self.off[a] + int(j) for a, j in zip(A, ia)] + [self.off[b] + int(j) for b, j in zip(B, ib)]
        coef = [1.0] * len(cols)
        if free is None:
            x = sum(self.knots[a][int(j)] for a, j in zip(A, ia)) + sum(self.knots[b][int(j)] for b, j in zip(B, ib))
            return cols, coef, max(x - self.K, 0.0)
        kf = self.knots[free]
        j = int(np.clip(np.searchsorted(kf, xf, side="right") - 1, 0, len(kf) - 2))
        th = (xf - kf[j]) / (kf[j + 1] - kf[j])
        cols += [self.off[free] + j, self.off[free] + j + 1]
        coef += [1.0 - th, th]
        return cols, coef, 0.0


def solve(knots, K, w, start=None, add=4000, seed=20000, max_pass=400, verbose=False):
    """Returns (V, d, v): LP value on the generated constraints, the smallest slack over
    all vertices and the solution.  ``start`` is a feasible portfolio (values at the
    knots); its tightest vertices seed the constraint set."""
    V = Vertices(knots, K)
    nv = V.off[-1]
    c = -np.concatenate(w)
    v = np.zeros(nv) if start is None else start.copy()
    # The hat functions of an asset sum to one, so a constant can be moved from
    # one h_i to another without changing anything.  Fix this gauge: h_i(0) = 0
    # for every asset but the first, which carries the cash component.
    bounds = [(-BIG, BIG)] * nv
    for i in range(1, V.n):
        v[V.off[0]:V.off[1]] += v[V.off[i]]
        v[V.off[i]:V.off[i + 1]] -= v[V.off[i]]
        bounds[V.off[i]] = (0.0, 0.0)
    rows, seen = [], set()
    value = np.nan
    for it in range(max_pass):
        if it == 0:
            d, items = V.worst(v, seed, only_violated=False)
            items = items + V.axis_items()
        else:
            d, items = V.worst(v, add)
            if d > -1e-9:
                if np.abs(v).max() > 0.99 * BIG:
                    raise RuntimeError("LP variables at the artificial bound")
                return value, d, v
        if verbose:
            print(f"    pass {it:3d}: constraints {len(rows):7d}  LP value {value:.9f}  smallest slack {d:.2e}", flush=True)
        new = 0
        for item in items:
            key = (item[1], tuple(item[4]), tuple(item[5]))
            if key not in seen:
                seen.add(key)
                rows.append(V.row(item))
                new += 1
        if new == 0:
            raise RuntimeError("no new constraint although a vertex is violated")
        indptr = np.concatenate([[0], np.cumsum([len(r[0]) for r in rows])])
        A = csr_matrix((np.concatenate([r[1] for r in rows]), np.concatenate([r[0] for r in rows]), indptr),
                       shape=(len(rows), nv))
        res = linprog(c, A_ub=A, b_ub=np.array([r[2] for r in rows]), bounds=bounds, method="highs",
                      options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
        if res.status != 0:
            raise RuntimeError(res.message)
        v, value = res.x, -res.fun
    raise RuntimeError("constraint generation did not converge")


def our_portfolio_value(strikes, knots, mu, s, K):
    """Our portfolio (kinks at its strikes, which are knots) priced under the truncated laws."""
    from subreplicating.subreplication import analytical_cash

    total = analytical_cash([np.asarray(x, float) for x in strikes], K)
    vals = []
    for i, k in enumerate(knots):
        f = k.copy()
        for j, st in enumerate(strikes[i]):
            f += (-1.0) ** (j + 1) * np.maximum(k - st, 0.0)
        vals.append(f)
        total += price_pwl(k, f, mu[i], s[i], 1 - TRUNC)
    vals[0] = vals[0] + analytical_cash([np.asarray(x, float) for x in strikes], K)   # cash on asset 1
    return total, vals


def main(argv):
    if argv[0] == "--check":
        d = json.load(open(HERE / "results" / f"{argv[1]}.json"))
        n = len(d["sig"])
        mu, s, _ = marginal_parameters(n)
        print(f"{argv[1]}: LP optimum on the knots used by their code  vs  the bound their code returned")
        for run in d["runs"]:
            if len(argv) > 2 and str(run["m"]) not in argv[2:]:
                continue
            knots = [np.asarray(k, float) for k in run["knots"]]
            t0 = time.time()
            start = [np.asarray(cf, float) for cf in run["coefs"]]
            start[0] = start[0] + float(run["constant"])
            V, dmin, _ = solve(knots, float(run["K"]), hat_prices(knots, mu, s), start=np.concatenate(start))
            print(f"  K = {run['K']:4d}  m = {run['m']:3d}  LP in [{V + min(dmin, 0):.9f}, {V:.9f}]  "
                  f"their LB = {run['LB']:.9f}  LP - LB = {V - run['LB']:+.2e}  [{time.time() - t0:.0f}s]", flush=True)
        return
    n = int(argv[0])
    ms = [int(x) for x in argv[1:]] or [4, 16]
    mu, s, ub = marginal_parameters(n)
    rows = json.load(open(HERE / "results" / f"final{n}.json"))
    out = []
    print(f"{n} assets: best portfolio with strikes on the augmented grid  vs  our portfolio")
    print("  (LP and 'ours, truncated' use the marginals truncated at the 1 - 1e-12 quantile; HV_ex the exact ones)")
    for r in rows:
        K = float(r["kb"])
        for m in ms:
            t0 = time.time()
            knots = build_knots(mu, s, ub, m, r["strikes_ex"])
            ours, vals = our_portfolio_value(r["strikes_ex"], knots, mu, s, K)
            V, dmin, _ = solve(knots, K, hat_prices(knots, mu, s), start=np.concatenate(vals))
            lo = V + min(dmin, 0.0)
            out.append({"K": K, "m": m, "knots_per_asset": [len(k) for k in knots], "lp_value": V,
                        "lp_value_feasible": lo, "hv_ex": r["hv_ex_star"], "hv_ex_truncated": ours,
                        "excess": V - ours, "seconds": time.time() - t0})
            print(f"  K = {K:4.0f}  m = {m:3d}  knots {[len(k) for k in knots]}  LP optimum in [{lo:.9f}, {V:.9f}]  "
                  f"ours, truncated = {ours:.9f}  (difference {V - ours:+.1e})  HV_ex = {r['hv_ex_star']:.9f}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            json.dump(out, open(HERE / "results" / f"grid_lp_{n}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
