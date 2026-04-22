"""Optional Numba-accelerated inner kernels used by ``polish_strikes`` at
larger basket sizes. Imported lazily; if Numba is not installed the pure-NumPy
path in ``polish_strikes`` is used instead.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange  # type: ignore[import-not-found]


@njit(cache=True, parallel=True, fastmath=True)
def fast_cash(
    idx_stack: np.ndarray,          # (V, n) int64
    candidates_flat: np.ndarray,    # concatenated candidate values per asset
    options_flat: np.ndarray,       # concatenated options values per asset
    cand_offsets: np.ndarray,       # (n+1,) start index of each asset's block
    kb: float,
) -> float:
    V, n = idx_stack.shape
    # numba's prange does not auto-detect a min-reduction via `if ... : best = ...`,
    # so we write per-vertex psi into an array (parallel, independent writes) and
    # reduce serially afterwards.
    psi = np.empty(V)
    for v in prange(V):
        s = 0.0
        opts = 0.0
        for i in range(n):
            k = idx_stack[v, i]
            off = cand_offsets[i]
            s += candidates_flat[off + k]
            opts += options_flat[off + k]
        basket = s - kb if s > kb else 0.0
        psi[v] = basket - s - opts
    return psi.min()


@njit(cache=True, fastmath=True)
def fast_v_delta(
    prices: np.ndarray,             # (N, n)
    strikes_flat: np.ndarray,       # concatenated strikes per asset
    strike_offsets: np.ndarray,     # (n+1,) — start of each asset's strike block
) -> float:
    """Contribution Σ_i Σ_j (-1)^(j+1) E[(S_i - K_ij)^+] to the portfolio value.

    Sign convention matches ``_compute_v``: j=0 ⇒ -1, j=1 ⇒ +1, ... alternating.
    """
    N, n = prices.shape
    inv_N = 1.0 / N
    total = 0.0
    for i in range(n):
        start = strike_offsets[i]
        end = strike_offsets[i + 1]
        K = end - start
        if K == 0:
            continue
        for j in range(K):
            sign = -1.0 if (j % 2) == 0 else 1.0  # (-1)^(j+1): j=0→-1, j=1→+1
            k = strikes_flat[start + j]
            acc = 0.0
            for row in range(N):
                d = prices[row, i] - k
                if d > 0.0:
                    acc += d
            total += sign * acc * inv_N
    return total
