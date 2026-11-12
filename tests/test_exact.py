"""Tests for the exact-marginal bounds (subreplicating.exact)."""

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtri

from subreplicating import lognormal_marginals
from subreplicating.exact import (
    black_call,
    hedging_value_exact,
    polish_strikes_exact,
    reassembled_basket_value,
)
from subreplicating.subreplication import port_value

T = 0.5


def test_black_call_matches_fine_discretisation():
    w, vol = 40.0, 0.3
    x = lognormal_marginals(np.array([w]), np.array([vol]), T, 2_000_000)[:, 0]
    for k in (20.0, 40.0, 70.0):
        assert abs(black_call(w, vol * np.sqrt(T), k) - np.maximum(x - k, 0).mean()) < 1e-4


def test_reassembly_of_antimonotone_pair_is_the_exact_antimonotone_coupling():
    w = np.array([50.0, 50.0])
    vols = np.array([0.4, 0.2])
    kb = 105.0
    n = 2_000
    p = lognormal_marginals(w, vols, T, n)
    p[:, 1] = p[::-1, 1]                      # antimonotone discrete coupling
    s = vols * np.sqrt(T)

    def payoff(u):
        x0 = w[0] * np.exp(s[0] * ndtri(u) - 0.5 * s[0] ** 2)
        x1 = w[1] * np.exp(s[1] * ndtri(1 - u) - 0.5 * s[1] ** 2)
        return max(x0 + x1 - kb, 0.0)

    pts = [1e-9, 1e-6, 1e-3, 0.5, 1 - 1e-3, 1 - 1e-6, 1 - 1e-9]
    exact = quad(payoff, 0, 1, points=pts, limit=500, epsabs=1e-12)[0]
    assert abs(reassembled_basket_value(p, w, vols, T, kb) - exact) < 1e-7


def test_bracket_is_ordered_and_polish_does_not_decrease():
    w = np.full(3, 100 / 3)
    vols = np.array([0.355, 0.20, 0.25])
    kb = 100.0
    p = lognormal_marginals(w, vols, T, 2_000)
    rng = np.random.default_rng(0)
    for i in range(3):
        p[:, i] = rng.permutation(p[:, i])    # an arbitrary coupling
    m = [np.array([13.0, 50.0]), np.array([25.0, kb]), np.array([24.0, 60.0])]
    hv_ex = hedging_value_exact(m, w, vols, T, kb)
    m_pol, hv_pol = polish_strikes_exact(m, w, vols, T, kb, unpark_at=p.max(axis=0))
    assert hv_pol >= hv_ex - 1e-12
    assert abs(hv_pol - hedging_value_exact(m_pol, w, vols, T, kb)) < 1e-12
    # weak duality: any sub-replicator is below any coupling of the exact marginals
    assert hv_pol <= reassembled_basket_value(p, w, vols, T, kb) + 1e-9
    # the exact price is close to the price on a fine discretisation
    fine = lognormal_marginals(w, vols, T, 400_000)
    assert abs(hv_ex - port_value(fine, m, kb).value) < 1e-3


def test_affine_and_mixture_marginals_are_consistent():
    from subreplicating.exact import AffineMarginal, TruncatedNormalMixtureMarginal

    base = TruncatedNormalMixtureMarginal([0.5, 0.5], [-2.0, 3.0], [1.5, 2.0], -10.0, 10.0)
    for a, b in ((0.7, 7.0), (-0.4, 4.0)):
        mg = AffineMarginal(base, a, b)
        x = mg.discretise(400_000)
        assert np.all(np.diff(x) >= 0)                       # quantiles are increasing
        assert abs(mg.mean - x.mean()) < 1e-4
        for k in (1.0, 5.0, 9.0):
            assert abs(mg.call(k) - np.maximum(x - k, 0).mean()) < 1e-4
        # slab(r, t, N) is the quantile at level (r + t) / N
        r, t, n = np.array([0.0, 3.0, 9.0]), 0.3, 10
        assert np.allclose(mg.slab(r, t, n), mg.quantile((r + t) / n), atol=1e-9)


def test_released_strikes_are_kept_only_above_the_release_gain():
    w = np.full(3, 100 / 3)
    vols = np.array([0.355, 0.20, 0.25])
    kb = 100.0
    p = lognormal_marginals(w, vols, T, 2_000)
    m = [np.array([13.0, 50.0]), np.array([25.0, kb]), np.array([24.0, 60.0])]
    base = hedging_value_exact(m, w, vols, T, kb)
    # a prohibitive threshold keeps the parked strike at kb ...
    m_parked, hv_parked = polish_strikes_exact(m, w, vols, T, kb, unpark_at=p.max(axis=0),
                                               release_gain=1.0)
    assert m_parked[1][-1] == kb
    # ... and no threshold can only do better; neither is below the start
    m_free, hv_free = polish_strikes_exact(m, w, vols, T, kb, unpark_at=p.max(axis=0),
                                           release_gain=0.0)
    assert hv_free >= hv_parked >= base
    assert abs(hv_free - hedging_value_exact(m_free, w, vols, T, kb)) < 1e-12
