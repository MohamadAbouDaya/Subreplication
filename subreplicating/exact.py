"""Bounds for the exact (non-discretised) marginals.

The rearrangement algorithms work on marginals discretised into ``N``
equiprobable atoms, so the equality ``HV = BV`` of the main pipeline certifies
optimality for the discretised problem only.  This module closes the gap to
the original problem from both sides:

* :func:`polish_strikes_exact` maximises the price of the sub-replicating
  portfolio under the exact marginals with the exact cash component.  The
  portfolio sub-replicates on the whole of R^n_+, so its price ``HV_ex`` is a
  lower bound for the original problem.
* :func:`reassembled_basket_value` turns the discrete coupling into a coupling
  of the exact marginals: the atom of rank ``r`` of asset ``i`` is replaced by
  the slab of the exact law between the quantile levels ``r/N`` and
  ``(r+1)/N``.  Its basket value ``BV_ex`` is an upper bound for the original
  problem.

A marginal is any object with the methods of :class:`LogNormalMarginal`
(``mean``, ``call``, ``quantile``, ``slab``) and the flag ``unbounded``.  The
functions also accept the parametrisation of :func:`lognormal_marginals`
(``weights, vols, T``) in place of a list of marginals.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import ndtr, ndtri

from .subreplication import analytical_cash, local_search


def black_call(weight: float, s: float, strike: float) -> float:
    """E[(X - strike)^+] for X = weight * exp(s Z - s^2 / 2), Z standard normal."""
    if strike <= 0.0:
        return float(weight - strike)
    d1 = (np.log(weight / strike) + 0.5 * s * s) / s
    return float(weight * ndtr(d1) - strike * ndtr(d1 - s))


class LogNormalMarginal:
    """X = weight * exp(s Z - s^2 / 2)."""

    unbounded = True

    def __init__(self, weight: float, s: float):
        self.weight = float(weight)
        self.s = float(s)

    @property
    def mean(self) -> float:
        return self.weight

    def call(self, strike: float) -> float:
        return black_call(self.weight, self.s, strike)

    def quantile(self, u):
        return self.weight * np.exp(self.s * ndtri(u) - 0.5 * self.s**2)

    def slab(self, r, t, n_rows: int):
        """Quantile at level (r + t) / n_rows, evaluated from the nearer tail."""
        r = np.asarray(r, float)
        lo = np.maximum((r + t) / n_rows, 1e-300)
        hi = np.maximum((n_rows - r - t) / n_rows, 1e-300)
        z = np.where(r < n_rows / 2, ndtri(np.minimum(lo, 0.75)), -ndtri(np.minimum(hi, 0.75)))
        return self.weight * np.exp(self.s * z - 0.5 * self.s**2)


class TruncatedNormalMixtureMarginal:
    """Mixture of normals truncated to [lb, ub], reported as X - shift.

    With ``shift = lb`` the variable lives on [0, ub - lb], which is the form
    the sub-replication construction expects (non-negative assets).
    """

    unbounded = False

    def __init__(self, weights, means, stds, lb: float, ub: float, shift: float = 0.0):
        self.w = np.atleast_1d(np.asarray(weights, float))
        self.w = self.w / self.w.sum()
        self.mu = np.atleast_1d(np.asarray(means, float))
        self.sd = np.atleast_1d(np.asarray(stds, float))
        self.lb, self.ub, self.shift = float(lb), float(ub), float(shift)
        self._cdf_lb = float(self._raw_cdf(np.array([self.lb]))[0])
        self._norm = float(self._raw_cdf(np.array([self.ub]))[0]) - self._cdf_lb

    def _raw_cdf(self, x):
        x = np.asarray(x, float)
        return (self.w * ndtr((x[..., None] - self.mu) / self.sd)).sum(axis=-1)

    def _partial(self, a: float, b: float, slope: float, intercept: float) -> float:
        """E[(slope * Y + intercept) 1{a < Y < b}] for the untruncated mixture Y."""
        za, zb = (a - self.mu) / self.sd, (b - self.mu) / self.sd
        mass = ndtr(zb) - ndtr(za)
        dens = (np.exp(-0.5 * za**2) - np.exp(-0.5 * zb**2)) / np.sqrt(2 * np.pi)
        return float(self.w @ ((slope * self.mu + intercept) * mass + slope * self.sd * dens))

    @property
    def mean(self) -> float:
        return self._partial(self.lb, self.ub, 1.0, -self.shift) / self._norm

    def call(self, strike: float) -> float:
        k = min(max(strike + self.shift, self.lb), self.ub)      # strike in the unshifted scale
        if strike + self.shift <= self.lb:
            return self.mean - strike
        return self._partial(k, self.ub, 1.0, -(strike + self.shift)) / self._norm

    def quantile(self, u):
        """Vectorised bisection on the mixture distribution function."""
        u = np.asarray(u, float)
        target = u * self._norm + self._cdf_lb
        lo = np.full(u.shape, self.lb)
        hi = np.full(u.shape, self.ub)
        for _ in range(52):
            mid = 0.5 * (lo + hi)
            right = self._raw_cdf(mid) < target
            lo = np.where(right, mid, lo)
            hi = np.where(right, hi, mid)
        return 0.5 * (lo + hi) - self.shift

    def slab(self, r, t, n_rows: int):
        return self.quantile((np.asarray(r, float) + t) / n_rows)

    def discretise(self, n: int) -> NDArray:
        """n equiprobable atoms at the mid-point quantiles."""
        return self.quantile(np.linspace(0.5 / n, 1 - 0.5 / n, n))


class AffineMarginal:
    """X = a * Y + b for a base marginal Y and a non-zero scale a (possibly negative)."""

    def __init__(self, base, a: float, b: float = 0.0):
        if a == 0.0:
            raise ValueError("the scale must be non-zero")
        self.base, self.a, self.b = base, float(a), float(b)
        self.unbounded = base.unbounded

    @property
    def mean(self) -> float:
        return self.a * self.base.mean + self.b

    def call(self, strike: float) -> float:
        c = (strike - self.b) / self.a
        if self.a > 0:
            return self.a * self.base.call(c)
        # (aY + b - k)^+ = |a| (c - Y)^+ and E[(c - Y)^+] = call_Y(c) - E[Y] + c
        return -self.a * (self.base.call(c) - self.base.mean + c)

    def quantile(self, u):
        u = np.asarray(u, float)
        return self.a * self.base.quantile(u if self.a > 0 else 1.0 - u) + self.b

    def slab(self, r, t, n_rows: int):
        if self.a > 0:
            return self.a * self.base.slab(r, t, n_rows) + self.b
        r = np.asarray(r, float)
        return self.a * self.base.slab(n_rows - 1 - r, 1.0 - t, n_rows) + self.b

    def discretise(self, n: int) -> NDArray:
        return self.quantile(np.linspace(0.5 / n, 1 - 0.5 / n, n))


def _as_marginals(marginals_or_weights, vols=None, T=None) -> Sequence:
    if vols is None:
        return list(marginals_or_weights)
    s = np.asarray(vols, float) * np.sqrt(T)
    return [LogNormalMarginal(w, si) for w, si in zip(marginals_or_weights, s, strict=True)]


def hedging_value_exact(m: list[NDArray], marginals, vols=None, T=None, kb: float | None = None) -> float:
    """Price of the portfolio with strikes ``m`` under the exact marginals.

    Same sign convention as ``port_value``: a long position in each asset and
    calls of alternating sign (short, long, ...) at the sorted strikes, plus
    the exact cash component of Proposition 2.3.  Call either as
    ``hedging_value_exact(m, marginals, kb=kb)`` or, for log-normal marginals,
    as ``hedging_value_exact(m, weights, vols, T, kb)``.
    """
    margs = _as_marginals(marginals, vols, T)
    v = float(sum(mg.mean for mg in margs))
    for i, strikes in enumerate(m):
        for j, k in enumerate(strikes):
            v += (-1.0) ** (j + 1) * margs[i].call(float(k))
    return v + analytical_cash(m, kb)


def polish_strikes_exact(
    m_init: list[NDArray],
    marginals,
    vols=None,
    T=None,
    kb: float | None = None,
    unpark_at: NDArray | None = None,
    xatol: float = 1e-8,
    fatol: float = 1e-12,
    max_rounds: int = 4,
    release_gain: float = 1e-6,
) -> tuple[list[NDArray], float]:
    """Local search on :func:`hedging_value_exact`, started from ``m_init``.

    Strikes at 0 are held fixed (they cancel the long position in the asset).
    The search (:func:`subreplicating.subreplication.local_search`, Nelder-Mead
    alternated with Powell) is run from two starting points:

    * ``m_init`` as it is, the strikes equal to ``kb`` staying there when
      ``unpark_at`` is given;
    * ``m_init`` with the strikes that the discretised pipeline parked at
      ``kb`` (the call was worthless on the sample) released: they start from
      ``unpark_at[i]`` (typically the largest atom of asset ``i``) and are
      free to move in ``(0, kb]``.

    The second result is kept only if it is worth more than the first by more
    than ``release_gain``.  A call struck that far out is worth so little that
    the value is almost flat in its strike; without the threshold the strike
    reported would depend on the optimiser's tolerances.

    Every strike configuration is a valid sub-replicator, so the returned
    value is a lower bound whatever the optimiser does, and it is never below
    the value at ``m_init``.
    """
    margs = _as_marginals(marginals, vols, T)
    eps = 1e-9
    m0 = [np.array(x, float) for x in m_init]
    best_m, best_v = m0, hedging_value_exact(m0, margs, kb=kb)

    # (starting strikes, mask of the strikes that are free to move)
    starts = [(m0, [(x > eps) & ((x < kb - eps) | (unpark_at is None)) for x in m0])]
    if unpark_at is not None and any((x >= kb - eps).any() for x in m0):
        released = []
        for i, x in enumerate(m0):
            x = x.copy()
            x[x >= kb - eps] = min(float(unpark_at[i]), kb - 1e-3)
            released.append(x)
        starts.append((released, [x > eps for x in released]))

    for start, free in starts:
        offsets = np.cumsum([0] + [int(f.sum()) for f in free])
        if offsets[-1] == 0:
            continue

        def rebuild(flat: NDArray, start=start, free=free, offsets=offsets) -> list[NDArray]:
            out = []
            for i, x in enumerate(start):
                y = x.copy()
                y[free[i]] = np.clip(flat[offsets[i]:offsets[i + 1]], eps, kb)
                out.append(np.sort(y))
            return out

        flat0 = np.concatenate([x[f] for x, f in zip(start, free, strict=True)])
        x_best, f_best = local_search(
            lambda f, rebuild=rebuild: -hedging_value_exact(rebuild(f), margs, kb=kb),
            flat0, xatol, fatol, maxiter=4000 * len(flat0), max_rounds=max_rounds,
        )
        # the first start is accepted on any improvement, the released one
        # only if it adds more than ``release_gain``
        threshold = best_v if start is starts[0][0] else best_v + release_gain
        if -f_best > threshold:
            best_m, best_v = rebuild(x_best), float(-f_best)
    return best_m, best_v


def reassembled_basket_value(
    prices_r: NDArray,
    marginals,
    vols=None,
    T=None,
    kb: float | None = None,
    nodes: int = 16,
    nodes_kink: int = 256,
) -> float:
    """Basket call value of the reassembly of ``prices_r`` with the exact marginals.

    ``prices_r`` is an (N, n) matrix whose columns are permutations of N
    equiprobable atoms (the output of RA / BRA).  Row ``j`` with ranks
    ``r_ij`` is replaced by the curve

        t -> ( F_i^{-1}((r_ij + t_i) / N) )_i ,   t_i = t or 1 - t,

    with one uniform ``t`` per row.  Since every rank occurs exactly once in
    each column, every coordinate is uniform on (0, 1) after the transform and
    the result is a coupling of the exact marginals, whatever orientation
    pattern (t or 1 - t per asset) is used in each row; the pattern giving the
    smallest value is selected row by row.

    The integral over ``t`` uses Gauss-Legendre quadrature with ``nodes``
    points, ``nodes_kink`` points for the rows on which the basket crosses the
    strike, and, for marginals with unbounded support, adaptive quadrature for
    the rows that contain the lowest or highest slab of some asset.
    """
    margs = _as_marginals(marginals, vols, T)
    n_rows, n = prices_r.shape
    ranks = np.argsort(np.argsort(prices_r, axis=0, kind="stable"), axis=0)
    patterns = [(1, *p) for p in itertools.product((1, -1), repeat=n - 1)]

    def row_values(rows: NDArray, g: int) -> tuple[NDArray, NDArray]:
        t, wt = np.polynomial.legendre.leggauss(g)
        t, wt = 0.5 * (t + 1.0), 0.5 * wt
        xp = [margs[i].slab(ranks[rows, i][:, None], t[None, :], n_rows) for i in range(n)]
        best = np.full(len(rows), np.inf)
        kink = np.zeros(len(rows), bool)
        for pat in patterns:
            tot = sum(xp[i] if pat[i] > 0 else xp[i][:, ::-1] for i in range(n)) - kb
            kink |= (tot.min(axis=1) < 0) & (tot.max(axis=1) > 0)
            best = np.minimum(best, np.maximum(tot, 0.0) @ wt)
        return best, kink

    rows = np.arange(n_rows)
    extreme = np.zeros(n_rows, bool)
    for i in range(n):
        if margs[i].unbounded:
            extreme |= (ranks[:, i] == 0) | (ranks[:, i] == n_rows - 1)
    vals = np.zeros(n_rows)
    regular = rows[~extreme]
    if len(regular) > 0:
        vals[regular], kink = row_values(regular, nodes)
        if kink.any():
            vals[regular[kink]], _ = row_values(regular[kink], nodes_kink)
    cuts = [0.0, 1e-6, 1e-3, 0.1, 0.9, 1 - 1e-3, 1 - 1e-6, 1.0]
    for j in rows[extreme]:
        best = np.inf
        for pat in patterns:
            def payoff(t, pat=pat, j=j):
                tot = sum(
                    margs[i].slab(ranks[j, i], t if pat[i] > 0 else 1.0 - t, n_rows)
                    for i in range(n)
                )
                return max(float(tot) - kb, 0.0)
            best = min(best, sum(
                quad(payoff, a, b, limit=200, epsabs=1e-14, epsrel=1e-10)[0]
                for a, b in zip(cuts[:-1], cuts[1:], strict=True)
            ))
        vals[j] = best
    return float(vals.mean())
