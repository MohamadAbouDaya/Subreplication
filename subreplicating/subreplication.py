"""Subreplicating portfolio construction for basket options.

Implements the Hobson-Laurence-Wang approach: given marginal distributions
(no dependence assumption), build a portfolio of vanilla calls on individual
assets that is always worth less than the basket call — a model-free lower
price bound.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

from .types import PortfolioValueResult, SubportfolioResult

# ---------------------------------------------------------------------------
# phi function  (fully vectorised — no Python loop)
# ---------------------------------------------------------------------------

def phi_calc(
    x: NDArray,
    y: NDArray,
    k: float,
    xpart: NDArray,
) -> NDArray:
    """Compute the phi function over a grid *xpart*.

    *x* and *y* must be **sorted ascending**.  *k* is the basket strike.
    Returns an array of the same length as *xpart*.

    The phi function determines optimal strike placement by evaluating
    d/dk [C_X(k1) + C_Y(k-k1)] along a partition of [0, k].
    """
    n = len(x)

    # x and y are sorted ascending (precondition), so ordinal ranks are 1..n.
    # The expression reduces to evenly-spaced grid points of the proxy CDF.
    ranks = np.arange(1, n + 1)
    d1_vals = 0.5 / n + (ranks - 1) / n - 1.0
    d2_vals = d1_vals  # same formula, same ranks

    # ------------------------------------------------------------------
    # Interpolate d1 at xpart and d2 at (k - xpart).
    # np.interp handles boundary clamping and exact matches robustly.
    # ------------------------------------------------------------------
    y_query = k - xpart

    interp1 = np.interp(xpart, x, d1_vals)
    interp2 = np.interp(y_query, y, d2_vals)

    return 1.0 + interp1 + interp2


# ---------------------------------------------------------------------------
# Strike finder  (vectorised zero-crossing detection)
# ---------------------------------------------------------------------------

def _find_crossings(
    phi: NDArray,
    xpart: NDArray,
    p1: NDArray,
    p2: NDArray,
    kb: float,
    min_gap: float | None = None,
) -> NDArray:
    """Find the interior zero-crossings of *phi* (merged and cleaned).

    Returns the merged crossing locations (without boundary padding).
    """
    if min_gap is None:
        min_gap = 0.0  # no merging by default; analytical cash handles all strike sets

    signs = np.sign(phi)
    sign_changes = np.where(np.diff(signs) != 0)[0]

    if len(sign_changes) == 0:
        return np.empty(0)

    p1_min, p1_max = p1.min(), p1.max()
    p2_min, p2_max = p2.min(), p2.max()

    valid = []
    for i in sign_changes:
        a, b = xpart[i], kb - xpart[i]
        if (a <= p1_min and b >= p2_max) or (a >= p1_max and b <= p2_min):
            continue
        valid.append(i)

    if len(valid) == 0:
        return np.empty(0)

    raw = xpart[np.array(valid)]

    # Merge nearby crossings
    clusters: list[list[float]] = [[raw[0]]]
    for k_val in raw[1:]:
        if k_val - clusters[-1][-1] < min_gap:
            clusters[-1].append(k_val)
        else:
            clusters.append([k_val])

    merged: list[float] = []
    for cl in clusters:
        if len(cl) % 2 == 1:
            merged.append(0.5 * (cl[0] + cl[-1]))

    return np.array(merged) if merged else np.empty(0)


def _build_strikes(crossings: NDArray, kb: float, phi0_positive: bool) -> NDArray:
    """Build strike array from crossings using the given parity convention."""
    if len(crossings) == 0:
        return np.empty(0)
    if phi0_positive and len(crossings) % 2 == 0:
        return np.concatenate(([0.0], crossings, [kb]))
    elif phi0_positive and len(crossings) % 2 == 1:
        return np.concatenate(([0.0], crossings))
    elif not phi0_positive and len(crossings) % 2 == 0:
        return crossings
    else:
        return np.concatenate((crossings, [kb]))


def k_finder(
    phi: NDArray,
    xpart: NDArray,
    p1: NDArray,
    p2: NDArray,
    kb: float,
    min_gap: float | None = None,
) -> NDArray:
    """Find optimal strikes from zero-crossings of *phi*.

    Uses the standard parity rule (phi[0] sign).  For the case where
    phi[0] is ambiguous, see ``sub_portfolio`` which optimises over
    both parities jointly across all assets.
    """
    crossings = _find_crossings(phi, xpart, p1, p2, kb, min_gap)
    if len(crossings) == 0:
        return np.empty(0)
    return _build_strikes(crossings, kb, phi[0] > 0)


# ---------------------------------------------------------------------------
# Cash component
# ---------------------------------------------------------------------------

def analytical_cash(m: list[NDArray], kb: float) -> float:
    """Compute the cash component exactly by exhaustive vertex enumeration.

    Ψ(x) = max(Σx_i − kb, 0) − Σ_i f_i(x_i)  is piecewise linear on R^n_+.
    Its minimum occurs at a vertex of the grid defined by each asset's strike
    breakpoints plus boundaries {0, kb}. Vectorised over all vertices.
    """
    n_assets = len(m)

    # Candidate x_i values per asset: {0} ∪ strikes_i ∪ {kb}
    candidates = [np.array(sorted(set([0.0, *m[i], kb]))) for i in range(n_assets)]

    # Build the (V, n_assets) matrix of all vertices via cartesian product.
    mesh = np.meshgrid(*candidates, indexing="ij")
    X = np.stack([g.ravel() for g in mesh], axis=1)       # (V, n_assets)
    s = X.sum(axis=1)                                      # (V,)
    basket = np.maximum(s - kb, 0.0)                       # (V,)

    options = np.zeros_like(s)
    for i in range(n_assets):
        strikes = m[i]
        if len(strikes) == 0:
            continue
        signs = (-1.0) ** (np.arange(len(strikes)) + 1)    # (K_i,)
        diffs = np.maximum(X[:, i, None] - strikes[None, :], 0.0)  # (V, K_i)
        options += diffs @ signs

    psi = basket - options - s
    return float(psi.min())


def monte_carlo_cash(
    m: list[NDArray],
    kb: float,
    n_mc: int,
    upper: float,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate the cash component via MC (kept for validation / fallback)."""
    if rng is None:
        rng = np.random.default_rng()

    n_assets = len(m)
    d1 = rng.uniform(0.0, upper, size=(n_mc, n_assets))

    option_payoffs = np.zeros(n_mc)
    for i in range(n_assets):
        strikes = m[i]
        for j in range(len(strikes)):
            sign = (-1.0) ** (j + 1)
            option_payoffs += sign * np.maximum(d1[:, i] - strikes[j], 0.0)

    basket_payoff = np.maximum(d1.sum(axis=1) - kb, 0.0)
    cash = float(np.min(basket_payoff - option_payoffs - d1.sum(axis=1)))
    return cash


# ---------------------------------------------------------------------------
# Full subreplication pipeline
# ---------------------------------------------------------------------------

def _compute_v(prices: NDArray, m: list[NDArray]) -> float:
    """Compute the vanilla-option component of the portfolio value.

    v = Σ_i [ E[S_i] + Σ_j (-1)^(j+1) E[max(S_i - K_ij, 0)] ]
    (the leading term is the k=0 call, which equals E[S_i]).
    """
    # Leading k=0 calls reduce to column means.
    v = float(prices.mean(axis=0).sum())

    n_rows = prices.shape[0]
    for i, strikes in enumerate(m):
        if len(strikes) == 0:
            continue
        signs = (-1.0) ** (np.arange(len(strikes)) + 1)
        payoffs = np.maximum(prices[:, i, None] - strikes[None, :], 0.0)  # (N, K_i)
        v += float((payoffs.sum(axis=0) / n_rows) @ signs)
    return v


def _quick_hv(prices: NDArray, m: list[NDArray], kb: float) -> float:
    """Deterministic HV estimate using analytical cash — no MC noise."""
    v = _compute_v(prices, m)
    cash = analytical_cash(m, kb)
    return v + cash


def _build_candidate_strikes(
    crossings_list: list[NDArray],
    prices: NDArray,
    kb: float,
) -> list[list[NDArray]]:
    """For each asset, build the two candidate strike arrays (parity True/False).

    Empty arrays are used when a parity yields fewer than 2 strikes.
    """
    n_assets = len(crossings_list)
    out: list[list[NDArray]] = []
    for i in range(n_assets):
        asset_max = float(prices[:, i].max())
        per_parity: list[NDArray] = []
        for p in (True, False):
            strikes = _build_strikes(crossings_list[i], kb, p).copy()
            if len(strikes) >= 2 and strikes[-1] < kb and strikes[-1] >= asset_max:
                strikes[-1] = kb
            if len(strikes) < 2:
                strikes = np.empty(0)
            per_parity.append(strikes)
        out.append(per_parity)
    return out


def _options_at(x: float, strikes: NDArray) -> float:
    """Σ_j (-1)^(j+1) max(x - strikes[j], 0) — matches analytical_cash sign rule."""
    if len(strikes) == 0:
        return 0.0
    signs = (-1.0) ** (np.arange(len(strikes)) + 1)
    return float((np.maximum(x - strikes, 0.0) * signs).sum())


def sub_portfolio(
    prices: NDArray,
    kb: float,
    n_partition: int = 100_000,
    n_mc: int = 1_000_000,
    upper: float | None = None,
    seed: int | None = None,
    top_k_parities: int = 64,
) -> SubportfolioResult:
    """Compute the subreplicating portfolio for a basket call with strike *kb*.

    For each asset the phi function is computed and its zero-crossings found.
    Because the parity rule (whether to prepend 0 / append kb to the strike
    list) depends on the sign of phi near the boundary — which is often
    numerically ambiguous for OTM options — we search over the 2^n parity
    combinations and keep the one that maximises HV = v + cash.

    Speedups vs. the naive 2^n loop:
      * The per-asset candidate set {0} ∪ crossings ∪ {kb} is invariant under
        the parity choice, so the exhaustive-vertex mesh is built exactly once.
      * HV = v + cash where v is additive across assets. We pre-compute
        v(asset, parity) in O(n·2) time, form v-totals for all 2^n parities in
        O(n·2^n) integer work, and only evaluate the expensive cash term for
        the top ``top_k_parities`` parities ranked by v. At n=10 this cuts the
        number of full cash evaluations from 1024 to 64 while preserving the
        global HV maximum in every tested case.

    Parameters as before; ``top_k_parities`` caps the number of parities for
    which the full analytical cash is evaluated.
    """
    del seed, n_mc, upper  # kept for API stability; no RNG use inside.
    n_rows, n_assets = prices.shape

    # --- phi and zero-crossings per asset ---------------------------------
    xgrid = np.linspace(kb / n_partition, kb, n_partition)
    phi_list: list[NDArray] = []
    crossings_list: list[NDArray] = []
    tiebreak = np.linspace(0.0, 1e-10, n_rows)

    for i in range(n_assets):
        p1 = np.sort(prices[:, i])
        other = prices[:, np.arange(n_assets) != i].sum(axis=1) + tiebreak
        p2 = np.sort(other)
        phi = phi_calc(p1, p2, kb, xgrid)
        phi_list.append(phi)
        crossings_list.append(_find_crossings(phi, xgrid, p1, p2, kb))

    # --- per-(asset, parity) candidate strike arrays ----------------------
    strikes_by_parity = _build_candidate_strikes(crossings_list, prices, kb)

    # If *any* asset has no valid strikes under either parity, fall back to
    # the empty-strike portfolio (matches the old behaviour).
    if any(all(len(s) == 0 for s in by_p) for by_p in strikes_by_parity):
        m_list = [np.empty(0)] * n_assets
        return SubportfolioResult(
            cash=analytical_cash(m_list, kb),
            v=_compute_v(prices, m_list),
            value=analytical_cash(m_list, kb) + _compute_v(prices, m_list),
            m=m_list,
            phi=phi_list,
        )

    # --- pre-compute v per (asset, parity) --------------------------------
    v_mean = float(prices.mean(axis=0).sum())   # leading k=0 call = Σ E[S_i]
    v_delta = np.zeros((n_assets, 2))           # contribution beyond the mean
    for i in range(n_assets):
        col = prices[:, i]
        for pb, strikes in enumerate(strikes_by_parity[i]):
            if len(strikes) == 0:
                v_delta[i, pb] = -np.inf  # parity invalid; exclude
                continue
            signs = (-1.0) ** (np.arange(len(strikes)) + 1)
            payoffs = np.maximum(col[:, None] - strikes[None, :], 0.0)
            v_delta[i, pb] = float((payoffs.mean(axis=0)) @ signs)

    # --- enumerate 2^n parities, rank by v, keep top-k --------------------
    n_parities = 1 << n_assets
    parity_bits = np.array(
        [[(p >> i) & 1 for i in range(n_assets)] for p in range(n_parities)],
        dtype=np.int64,
    )  # shape (2^n, n_assets); 0 → parity True, 1 → parity False

    v_totals = v_mean + v_delta[np.arange(n_assets), parity_bits].sum(axis=1)
    valid = np.isfinite(v_totals)
    if not valid.any():
        m_list = [np.empty(0)] * n_assets
        return SubportfolioResult(
            cash=analytical_cash(m_list, kb),
            v=_compute_v(prices, m_list),
            value=analytical_cash(m_list, kb) + _compute_v(prices, m_list),
            m=m_list,
            phi=phi_list,
        )

    k = min(top_k_parities, int(valid.sum()))
    order = np.argsort(-np.where(valid, v_totals, -np.inf))[:k]

    # --- build the shared vertex mesh once --------------------------------
    candidates = [
        np.array(sorted({0.0, *crossings_list[i], kb})) for i in range(n_assets)
    ]
    cand_idx_mesh = np.meshgrid(
        *[np.arange(len(c)) for c in candidates], indexing="ij"
    )
    idx_stack = np.stack([g.ravel() for g in cand_idx_mesh], axis=1)  # (V, n)
    V = idx_stack.shape[0]

    X = np.empty_like(idx_stack, dtype=float)
    for i in range(n_assets):
        X[:, i] = candidates[i][idx_stack[:, i]]
    s = X.sum(axis=1)
    basket_minus_s = np.maximum(s - kb, 0.0) - s  # (V,)

    # options_table[i]: (2, c_i) — options term for asset i at each candidate
    options_table: list[NDArray] = []
    for i in range(n_assets):
        tbl = np.zeros((2, len(candidates[i])))
        for pb, strikes in enumerate(strikes_by_parity[i]):
            if len(strikes) == 0:
                tbl[pb, :] = -np.inf  # marker; excluded by v-ranking above
                continue
            for k_idx, x in enumerate(candidates[i]):
                tbl[pb, k_idx] = _options_at(float(x), strikes)
        options_table.append(tbl)

    # --- evaluate cash for the top-k parities in one batched sweep --------
    # options_sum has shape (V, k); we add each asset's contribution via a
    # broadcasting gather parameterised by the asset's top-k parity bits.
    bits_top = parity_bits[order]                     # (k, n_assets)
    options_sum = np.zeros((V, len(order)))
    for i in range(n_assets):
        # contrib[v, p] = options_table[i][bits_top[p, i], idx_stack[v, i]]
        contrib = options_table[i][bits_top[:, i][None, :], idx_stack[:, i][:, None]]
        options_sum += contrib
    cash_per_parity = (basket_minus_s[:, None] - options_sum).min(axis=0)  # (k,)
    hv_per_parity = cash_per_parity + v_totals[order]

    best_local = int(np.argmax(hv_per_parity))
    best_bits = bits_top[best_local]
    best_m = [strikes_by_parity[i][best_bits[i]].copy() for i in range(n_assets)]

    cash = float(cash_per_parity[best_local])
    v = float(v_totals[order[best_local]])
    return SubportfolioResult(
        cash=cash,
        v=v,
        value=cash + v,
        m=best_m,
        phi=phi_list,
    )


def port_value(
    prices: NDArray,
    m: list[NDArray],
    kb: float,
) -> PortfolioValueResult:
    """Recompute portfolio value with (possibly manually adjusted) strikes *m*."""
    cash = analytical_cash(m, kb)
    v = _compute_v(prices, m)
    return PortfolioValueResult(cash=cash, v=v, value=cash + v)


# ---------------------------------------------------------------------------
# Local strike polish
# ---------------------------------------------------------------------------

try:  # Optional JIT inner kernels.
    from ._fast import fast_cash as _fast_cash_jit
    from ._fast import fast_v_delta as _fast_v_delta_jit
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - numba is optional
    _NUMBA_AVAILABLE = False


def polish_strikes(
    prices: NDArray,
    m_init: list[NDArray],
    kb: float,
    xatol: float = 1e-5,
    fatol: float = 1e-8,
    maxiter: int | None = None,
    use_numba: bool | None = None,
) -> tuple[list[NDArray], float]:
    """Locally polish strike values to maximise HV for a fixed coupling.

    HV(m) is continuous and piecewise-linear in the strike values, so a
    gradient-free local optimiser (Nelder-Mead) can close (a) the finite-grid
    discretisation error of the phi-crossing strikes and (b) the misalignment
    between the variance objective used inside BRA and the HV surface.

    The inner evaluation re-uses a fixed integer vertex-index mesh (rebuilt
    only if the candidate cardinality changes), avoiding the ~1M-vertex mesh
    allocation that the generic ``analytical_cash`` would otherwise repeat on
    every Nelder-Mead step. This makes the polish tractable at n ≥ 10.
    """
    from scipy.optimize import minimize

    n_assets = prices.shape[1]
    n_rows = prices.shape[0]
    inv_n_rows = 1.0 / n_rows
    col_means_sum = float(prices.mean(axis=0).sum())

    # --- freeze boundary strikes at exact 0 / kb ----------------------------
    # During polish the candidate set per asset = sorted({0} ∪ strikes ∪ {kb}).
    # If a strike at 0 or kb drifts into the interior the candidate cardinality
    # grows, which blows up the vertex mesh exponentially. Strikes placed at
    # the boundary by the phi-crossing construction are parity markers that
    # should not move; we hold them fixed and only optimise interior strikes.
    BOUNDARY_EPS = 1e-9
    interior_mask: list[NDArray] = []
    frozen_init: list[NDArray] = []
    for i in range(n_assets):
        mi = m_init[i]
        mask = np.ones(len(mi), dtype=bool)
        for j in range(len(mi)):
            if mi[j] <= BOUNDARY_EPS or mi[j] >= kb - BOUNDARY_EPS:
                mask[j] = False
        interior_mask.append(mask)
        frozen_init.append(mi.copy())

    free_sizes = [int(mask.sum()) for mask in interior_mask]
    starts = np.cumsum([0, *free_sizes])
    flat0 = np.concatenate(
        [frozen_init[i][interior_mask[i]] for i in range(n_assets)]
    ) if sum(free_sizes) > 0 else np.empty(0)

    if len(flat0) == 0:
        # Nothing movable — just report HV at the input strikes.
        return [x.copy() for x in m_init], float(port_value(prices, m_init, kb).value)

    if maxiter is None:
        # Nelder-Mead typically needs O(200 * d) evaluations.
        maxiter = 300 * len(flat0) + 2_000

    want_numba = _NUMBA_AVAILABLE if use_numba is None else (use_numba and _NUMBA_AVAILABLE)

    cache: dict[tuple[int, ...], NDArray] = {}

    def _idx_stack(cand_sizes: tuple[int, ...]) -> NDArray:
        hit = cache.get(cand_sizes)
        if hit is not None:
            return hit
        mesh = np.meshgrid(*[np.arange(c) for c in cand_sizes], indexing="ij")
        stack = np.stack([g.ravel() for g in mesh], axis=1).astype(np.int64)
        cache[cand_sizes] = stack
        return stack

    prices_c = np.ascontiguousarray(prices)  # numba prefers contiguous

    def _rebuild_m(flat: NDArray) -> list[NDArray]:
        """Reconstruct the full strike list: frozen boundary strikes keep their
        value; free (interior) strikes are read from ``flat`` and clipped into
        (BOUNDARY_EPS, kb - BOUNDARY_EPS) to keep candidate cardinality stable.
        """
        out: list[NDArray] = []
        for i, (a, b, mask) in enumerate(
            zip(starts[:-1], starts[1:], interior_mask, strict=True)
        ):
            mi = frozen_init[i].copy()
            if mask.any():
                mi[mask] = np.clip(flat[a:b], BOUNDARY_EPS, kb - BOUNDARY_EPS)
            out.append(np.sort(mi))
        return out

    def neg_hv(flat: NDArray) -> float:
        m = _rebuild_m(flat)
        candidates = [np.array(sorted({0.0, *m[i].tolist(), kb})) for i in range(n_assets)]
        cand_sizes = tuple(len(c) for c in candidates)
        idx_stack = _idx_stack(cand_sizes)

        # Per-asset options table (value at each candidate)
        options_per_asset: list[NDArray] = []
        for i in range(n_assets):
            K_i = len(m[i])
            if K_i == 0:
                options_per_asset.append(np.zeros(len(candidates[i])))
                continue
            signs = (-1.0) ** (np.arange(K_i) + 1)
            options_per_asset.append(
                np.maximum(candidates[i][:, None] - m[i][None, :], 0.0) @ signs
            )

        if want_numba:
            cand_offsets = np.zeros(n_assets + 1, dtype=np.int64)
            for i in range(n_assets):
                cand_offsets[i + 1] = cand_offsets[i] + len(candidates[i])
            candidates_flat = np.concatenate(candidates).astype(np.float64)
            options_flat = np.concatenate(options_per_asset).astype(np.float64)

            strike_offsets = np.zeros(n_assets + 1, dtype=np.int64)
            for i in range(n_assets):
                strike_offsets[i + 1] = strike_offsets[i] + len(m[i])
            strikes_flat = (
                np.concatenate(m).astype(np.float64) if any(len(mm) for mm in m)
                else np.zeros(0)
            )

            cash = float(
                _fast_cash_jit(idx_stack, candidates_flat, options_flat, cand_offsets, kb)
            )
            v_delta_sum = (
                float(_fast_v_delta_jit(prices_c, strikes_flat, strike_offsets))
                if len(strikes_flat) > 0 else 0.0
            )
            v = col_means_sum + v_delta_sum
        else:
            V = idx_stack.shape[0]
            s = np.zeros(V)
            options_sum = np.zeros(V)
            for i in range(n_assets):
                col_idx = idx_stack[:, i]
                s += candidates[i][col_idx]
                options_sum += options_per_asset[i][col_idx]
            cash = float((np.maximum(s - kb, 0.0) - s - options_sum).min())

            v = col_means_sum
            for i in range(n_assets):
                K_i = len(m[i])
                if K_i == 0:
                    continue
                signs = (-1.0) ** (np.arange(K_i) + 1)
                payoffs = np.maximum(prices[:, i, None] - m[i][None, :], 0.0)
                v += float((payoffs.sum(axis=0) * inv_n_rows) @ signs)

        return -(cash + v)

    res = minimize(
        neg_hv, flat0, method="Nelder-Mead",
        options={"xatol": xatol, "fatol": fatol, "maxiter": maxiter},
    )
    m_best = _rebuild_m(res.x)
    return m_best, float(-res.fun)
