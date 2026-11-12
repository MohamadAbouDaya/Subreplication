import numpy as np
import pytest
from scipy.stats import norm

from subreplicating.pricing import call_price, put_price
from subreplicating.rearrangement import bra
from subreplicating.subreplication import (
    analytical_cash,
    monte_carlo_cash,
    phi_calc,
    port_value,
    sub_portfolio,
)


def _lognormal_marginals(weights, vols, T, n):
    u = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    z = norm.ppf(u)[:, None]
    vols_row = vols[None, :]
    return weights[None, :] * np.exp(-vols_row**2 * T / 2 + vols_row * np.sqrt(T) * z)


def test_call_put_parity():
    rng = np.random.default_rng(0)
    s = rng.lognormal(mean=0, sigma=0.3, size=20_000)
    k = float(np.exp(0))
    # E[max(S-K,0)] - E[max(K-S,0)] = E[S] - K
    assert call_price(s, k) - put_price(s, k) == pytest.approx(s.mean() - k, abs=1e-10)


def test_call_price_at_zero_strike_equals_mean():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    assert call_price(s, 0.0) == pytest.approx(s.mean())


def test_phi_calc_at_boundaries():
    # phi(k=0): interp1(0)=d1[0], interp2(kb)=d2[-1]
    x = np.linspace(1.0, 10.0, 10)
    y = np.linspace(2.0, 12.0, 10)
    kb = 5.0
    grid = np.linspace(0.0, kb, 21)
    phi = phi_calc(x, y, kb, grid)
    assert phi.shape == grid.shape
    assert np.isfinite(phi).all()


def test_analytical_cash_matches_monte_carlo_two_assets():
    # Two-asset case with explicit strike sets — cheap & exhaustive cash should
    # agree with a high-resolution MC search to within MC noise.
    m = [np.array([0.0, 30.0, 70.0, 100.0]), np.array([0.0, 40.0, 80.0, 100.0])]
    kb = 100.0
    a = analytical_cash(m, kb)
    rng = np.random.default_rng(0)
    mc = monte_carlo_cash(m, kb, n_mc=200_000, upper=kb, rng=rng)
    # MC is a min over samples → upper bound on the true min, so mc >= a in
    # expectation. Allow a generous tolerance for finite-sample slack.
    assert mc + 0.5 >= a
    assert a == pytest.approx(mc, abs=1.5)


def test_analytical_cash_empty_strikes():
    # Ψ minimum over the box {0,kb}^n with no internal strikes:
    #   Ψ(x) = max(s-kb,0) - s. Vertices: each x_i in {0,kb}.
    # If all zero: 0. If k of them = kb: max((k-1)*kb, 0) - k*kb. Min over k.
    kb = 10.0
    n_assets = 3
    m = [np.empty(0)] * n_assets
    a = analytical_cash(m, kb)
    # k=0 gives 0; k=1 gives -kb; k=2 gives kb - 2*kb = -kb; k=3 gives 2kb - 3kb = -kb
    assert a == pytest.approx(-kb)


def test_sub_portfolio_value_matches_port_value():
    # port_value with the strikes returned by sub_portfolio must reproduce
    # the same value (cash + v) — otherwise the API is inconsistent.
    weights = np.array([100 / 3, 100 / 3, 100 / 3])
    vols = np.array([0.355, 0.20, 0.25])
    prices = _lognormal_marginals(weights, vols, T=0.5, n=2_000)
    r = bra(prices, maxiter=2_000, stalliter=3, seed=0)
    sr = sub_portfolio(r.X, kb=110.0, n_partition=5_000, seed=0)
    pv = port_value(r.X, sr.m, kb=110.0)
    assert pv.cash == pytest.approx(sr.cash)
    assert pv.v == pytest.approx(sr.v)
    assert pv.value == pytest.approx(sr.value)


def test_sub_portfolio_lower_bound_on_basket_call():
    # The whole point of subreplication: portfolio value should not exceed the
    # basket call price under the rearranged coupling.
    weights = np.array([100 / 3, 100 / 3, 100 / 3])
    vols = np.array([0.355, 0.20, 0.25])
    prices = _lognormal_marginals(weights, vols, T=0.5, n=2_000)
    r = bra(prices, maxiter=2_000, stalliter=3, seed=0)
    kb = 110.0
    sr = sub_portfolio(r.X, kb=kb, n_partition=5_000, seed=0)
    basket = prices.sum(axis=1)
    bv = call_price(basket, kb)
    # HV is constructed to lower-bound BV; allow a tiny numerical margin.
    assert sr.value <= bv + 1e-6


# ---------------------------------------------------------------------------
# Parity search, crossing filter and polish
# ---------------------------------------------------------------------------

def _candidate_strikes_like_sub_portfolio(prices, kb, n_partition):
    """Re-derive the per-(asset, parity) candidate strikes exactly as
    ``sub_portfolio`` does, for an independent brute-force comparison."""
    from subreplicating.subreplication import _build_candidate_strikes, _find_crossings

    n_rows, n_assets = prices.shape
    xgrid = np.linspace(kb / n_partition, kb, n_partition)
    tiebreak = np.linspace(0.0, 1e-10, n_rows)
    crossings = []
    for i in range(n_assets):
        p1 = np.sort(prices[:, i])
        p2 = np.sort(prices[:, np.arange(n_assets) != i].sum(axis=1) + tiebreak)
        phi = phi_calc(p1, p2, kb, xgrid)
        crossings.append(_find_crossings(phi, xgrid, p1, p2, kb))
    return _build_candidate_strikes(crossings, prices, kb)


def test_parity_search_matches_brute_force_seven_assets():
    # 2^7 = 128 parities.  The pruned search must return the global HV maximum
    # (a "top-k by option value" shortcut misses it badly).
    import itertools

    rng = np.random.default_rng(3)
    n_assets = 7
    weights = np.full(n_assets, 100.0 / n_assets)
    vols = rng.uniform(0.15, 0.9, size=n_assets)
    prices = _lognormal_marginals(weights, vols, T=0.5, n=1_000)
    r = bra(prices, maxiter=3_000, stalliter=50, seed=1)
    kb = 105.0
    n_partition = 5_000

    sr = sub_portfolio(r.X, kb, n_partition=n_partition, parity_batch=8)

    by_parity = _candidate_strikes_like_sub_portfolio(r.X, kb, n_partition)
    best = -np.inf
    for bits in itertools.product((0, 1), repeat=n_assets):
        m = [by_parity[i][b] for i, b in enumerate(bits)]
        if any(len(s) == 0 for s in m):
            continue
        best = max(best, port_value(r.X, m, kb).value)
    assert np.isfinite(best)
    # sub_portfolio never returns less than the two strike-free portfolios
    # (linear, and one call struck at kb per asset)
    linear = port_value(r.X, [np.empty(0)] * n_assets, kb).value
    otm = port_value(r.X, [np.array([0.0, kb])] * n_assets, kb).value
    assert sr.value == pytest.approx(max(best, linear, otm), abs=1e-9)


def test_find_crossings_ignores_exact_zero_plateau():
    from subreplicating.subreplication import _find_crossings

    x = np.linspace(1.0, 10.0, 100)
    y = np.linspace(1.0, 10.0, 100)
    kb = 5.0
    grid = np.linspace(0.05, kb, 100)
    # negative, then an exact-zero plateau up to the end of the grid: this is
    # the clamped region beyond the sample and must not create a crossing.
    phi_plateau = np.where(grid < 3.0, -0.1, 0.0)
    assert len(_find_crossings(phi_plateau, grid, x, y, kb)) == 0
    # a genuine -/+ change, possibly through an exact zero, counts exactly once
    phi_cross = np.where(grid < 3.0, -0.1, 0.1)
    phi_cross[int(np.argmax(grid >= 3.0))] = 0.0
    assert len(_find_crossings(phi_cross, grid, x, y, kb)) == 1


def test_polish_keeps_pointwise_subreplication_and_never_lowers_hv():
    from subreplicating.subreplication import polish_strikes

    weights = np.array([100 / 3, 100 / 3, 100 / 3])
    vols = np.array([0.355, 0.20, 0.25])
    prices = _lognormal_marginals(weights, vols, T=0.5, n=2_000)
    r = bra(prices, maxiter=5_000, stalliter=100, seed=0)
    kb = 110.0
    sr = sub_portfolio(r.X, kb=kb, n_partition=5_000)
    m, hv = polish_strikes(r.X, sr.m, kb, use_numba=False)

    # (a) polish never lowers HV and reports a value consistent with port_value
    assert hv >= sr.value - 1e-12
    pv = port_value(r.X, m, kb)
    assert pv.value == pytest.approx(hv, abs=1e-9)
    # (b) weak duality: HV <= BV under the coupling
    bv = call_price(r.X.sum(axis=1), kb)
    assert hv <= bv + 1e-9
    # (c) the polished portfolio is still a pointwise sub-replicator
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.5 * kb, size=(200_000, 3))
    port = np.full(len(X), pv.cash)
    for i, strikes in enumerate(m):
        port += X[:, i]
        for j, k in enumerate(strikes):
            port += (-1.0) ** (j + 1) * np.maximum(X[:, i] - k, 0.0)
    payoff = np.maximum(X.sum(axis=1) - kb, 0.0)
    assert (port <= payoff + 1e-9).all()


def test_degenerate_couplings_return_the_optimal_trivial_portfolio():
    """Jointly mixable marginals: the basket is constant, so the linear portfolio
    (strike below the mean) or the out-of-the-money one (strike above) is optimal."""
    import numpy as np

    from subreplicating import call_price, polish_strikes, sub_portfolio

    u = (np.arange(4000) + 0.5) / 4000
    # three uniforms on [0, 30] arranged so that every row sums to 45
    x1 = 30 * u
    x2 = 30 * ((u + 0.5) % 1.0)
    x3 = 45 - x1 - x2
    prices = np.column_stack([x1, x2, x3])
    assert np.allclose(prices.sum(axis=1), 45.0)
    for kb in (30.0, 45.0, 60.0):
        sr = sub_portfolio(prices, kb, n_partition=10_000)
        bv = call_price(prices.sum(axis=1), kb)
        assert abs(sr.value - max(45.0 - kb, 0.0)) < 1e-9
        assert abs(sr.value - bv) < 1e-9                  # optimal by weak duality
        m, hv = polish_strikes(prices, sr.m, kb)
        assert abs(hv - sr.value) < 1e-9


def test_refinement_is_applied_before_the_comparison_with_the_trivial_portfolios():
    """Deep out of the money the unrefined phi-crossing strikes can be worth less
    than the strike-free portfolios although, refined, they attain the basket
    value.  ``sub_portfolio`` must keep them (``m_search``) and
    ``refine_portfolio`` must polish them before comparing (regression test:
    they used to be discarded, and the pipeline returned HV = 0)."""
    from subreplicating.subreplication import refine_portfolio

    weights = np.array([100 / 3, 100 / 3, 100 / 3])
    vols = np.array([0.355, 0.20, 0.25])
    prices = _lognormal_marginals(weights, vols, T=0.5, n=2_000)
    r = bra(prices, maxiter=20_000, stalliter=200, seed=0)
    kb = 112.0
    sr = sub_portfolio(r.X, kb, n_partition=20_000)
    bv = call_price(r.X.sum(axis=1), kb)

    # unrefined: the strike search loses to the strike-free portfolio ...
    assert sr.value_search is not None and sr.m_search is not None
    assert sr.value_search < sr.value
    # ... refined: it is worth strictly more and attains the basket value
    m, hv = refine_portfolio(r.X, sr, kb)
    assert hv > sr.value + 1e-4
    assert hv <= bv + 1e-9
    assert hv == pytest.approx(bv, abs=1e-5)
    assert port_value(r.X, m, kb).value == pytest.approx(hv, abs=1e-9)


def test_selection_key_prefers_certified_then_strike_search():
    from subreplicating.examples._common import selection_key
    from subreplicating.types import SubportfolioResult

    X = np.array([[1.0, 2.0], [2.0, 1.0]])           # basket constant at 3
    kb = 2.0                                          # basket value 1
    certified = SubportfolioResult(cash=-2.0, v=3.0, value=1.0, m=[np.empty(0)] * 2, phi=[])
    searched = SubportfolioResult(cash=0.0, v=0.0, value=0.0, m=[np.empty(0)] * 2, phi=[],
                                  m_search=[np.array([0.5, 1.5])] * 2, value_search=-0.2)
    plain = SubportfolioResult(cash=0.0, v=0.0, value=0.0, m=[np.empty(0)] * 2, phi=[])
    keys = [selection_key(X, s, kb) for s in (certified, searched, plain)]
    assert keys[0] > keys[1] > keys[2]
    assert keys[1] == (1, -0.2)
