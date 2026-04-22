"""Rearrangement algorithms for finding worst-case dependence structures.

Implements RA (Rearrangement Algorithm), BRA (Block Rearrangement Algorithm),
and BRAVE (BRA with Variance Equalization) from the quantitative risk /
derivatives-pricing literature.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .types import RearrangementResult


def _variance_of_sums(s: NDArray) -> float:
    """Default objective: unbiased variance of the row-sum vector."""
    return float(np.var(s, ddof=1))


def basket_call_objective(kb: float) -> Callable[[NDArray], float]:
    """Objective that directly targets the basket-call lower bound at strike kb.

    Minimising E[(Σs - kb)^+] under the rearranged coupling is equivalent to
    finding the BRA-approximated lower bound BV for that strike. Because the
    payoff is convex in Σs, the supermodular rearrangement inequality still
    makes each counter-monotonic step of RA/BRA non-increasing in this
    objective (Rüschendorf, 1983). Useful when tightening the gap for a
    specific strike rather than a variance proxy.
    """
    def f(s: NDArray) -> float:
        return float(np.maximum(s - kb, 0.0).mean())
    return f


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def shuffle_matrix(X: NDArray, rng: np.random.Generator) -> NDArray:
    """Independently permute each column of *X* (vectorised)."""
    return rng.permuted(X, axis=0)


def _rank_descending(v: NDArray) -> NDArray:
    """Return 1-based ranks of *v* in **descending** order (ties broken by position)."""
    # argsort of -v gives indices that would sort descending;
    # a second argsort inverts to ranks.  +1 for 1-based.
    return np.argsort(np.argsort(-v, kind="mergesort"), kind="mergesort") + 1


def rearrange(block1: NDArray, block2: NDArray) -> NDArray:
    """Counter-monotonically rearrange *block1* against *block2*.

    Rows of *block1* are permuted so that its row-sums are sorted opposite to
    the row-sums of *block2* (high sums of block1 go with low sums of block2).
    """
    sums2 = block2.sum(axis=1) if block2.ndim == 2 else block2
    rank_b2 = _rank_descending(sums2)

    sums1 = block1.sum(axis=1) if block1.ndim == 2 else block1
    idx_sorted = np.argsort(sums1, kind="mergesort")

    # Largest block2 sum (rank 1) pairs with smallest block1 (idx_sorted[0]).
    perm = idx_sorted[rank_b2 - 1]
    return block1[perm]


def rearrange_partition(X: NDArray, partition: NDArray) -> NDArray:
    """Split columns of *X* by boolean *partition* and rearrange the smaller block."""
    d = X.shape[1]
    # Ensure the True-block is the smaller one
    if partition.sum() > d / 2:
        partition = ~partition

    if partition.sum() == 0:
        return X  # degenerate partition, nothing to do

    X = X.copy()
    block1 = X[:, partition]
    block2 = X[:, ~partition]
    X[:, partition] = rearrange(block1, block2)
    return X


# ---------------------------------------------------------------------------
# Greedy number partitioning  (used by equalvar / BRAVE)
# ---------------------------------------------------------------------------

def _greedy_partition(values: NDArray) -> tuple[float, NDArray]:
    """Greedy differencing algorithm for the number-partitioning problem.

    Returns (absolute difference, boolean partition mask).
    """
    n = len(values)
    if n == 0:
        return 0.0, np.empty(0, dtype=bool)

    idx_sorted = np.argsort(-np.abs(values))  # descending by |value|
    S = values[idx_sorted]

    partition = np.empty(n, dtype=bool)
    sum_a, sum_b = 0.0, 0.0

    for i in range(n):
        if (S[i] > 0) ^ (sum_a > sum_b):
            sum_a += S[i]
            partition[idx_sorted[i]] = True
        else:
            sum_b += S[i]
            partition[idx_sorted[i]] = False

    if sum_a < sum_b:
        partition = ~partition

    return abs(sum_a - sum_b), partition


def _equalvar(X: NDArray, partition_prev: NDArray, rng: np.random.Generator) -> NDArray:
    """Choose a column partition that approximately equalises block variances.

    Uses the covariance of each column with the total row-sum to decide which
    columns should be grouped together.
    """
    d = X.shape[1]
    # Ensure the True-block (prev) is the smaller one
    if partition_prev.sum() > d / 2:
        partition_prev = ~partition_prev

    demean = X - X.mean(axis=0, keepdims=True)
    # Covariance of each column with the total row-sum (un-normalised is fine for partitioning)
    covars = demean.T @ demean.sum(axis=1) / (X.shape[0] - 1)

    diff_a, part_a = _greedy_partition(covars[partition_prev])
    # Prepend the difference of the first block as an extra element
    combined = np.concatenate(([diff_a], covars[~partition_prev]))
    diff_b, part_b = _greedy_partition(combined)

    partition = np.zeros(d, dtype=bool)

    if not partition_prev.any():
        # First block was empty — take the partition of the larger block
        partition[:] = part_b
    else:
        # Combine: if the diff element (index 0 of part_b) is in True-block,
        # keep part_a as-is; otherwise invert it.
        if part_b[0]:
            partition[partition_prev] = part_a
        else:
            partition[partition_prev] = ~part_a
        partition[~partition_prev] = part_b[1:]

    return partition


# ---------------------------------------------------------------------------
# Sort result helper
# ---------------------------------------------------------------------------

def _sort_and_pack(
    X: NDArray,
    niter: int,
    fval: float,
    fiter: NDArray,
    converged: bool,
    **extra,
) -> RearrangementResult:
    S = X.sum(axis=1)
    order = np.argsort(-S, kind="mergesort")
    return RearrangementResult(
        X=X[order],
        S=S[order],
        niter=niter,
        fval=fval,
        fiter=fiter[:niter],
        converged=converged,
        **extra,
    )


# ---------------------------------------------------------------------------
# RA  —  Rearrangement Algorithm
# ---------------------------------------------------------------------------

def ra(
    X: NDArray,
    f: Callable[[NDArray], float] = _variance_of_sums,
    shuffle: bool = True,
    maxiter: int = 1_000,
    stalliter: int | None = None,
    abs_tol: float = 0.0,
    rel_tol: float = 0.0,
    f_target: float = -np.inf,
    seed: int | None = None,
) -> RearrangementResult:
    """Rearrangement Algorithm: cycle through columns, oppositely sorting each
    against the sum of all others to minimise *f(row_sums)*.
    """
    rng = np.random.default_rng(seed)
    X = X.copy()
    d = X.shape[1]
    if stalliter is None:
        stalliter = d
    if shuffle:
        X = shuffle_matrix(X, rng)

    fiter = np.empty(maxiter)
    fval = f(X.sum(axis=1))
    fiter[0] = fval
    niter = 1
    converged = False
    col = 0

    while fval > f_target and niter < maxiter and not converged:
        other_sums = X.sum(axis=1) - X[:, col]
        rank_desc = _rank_descending(other_sums)      # 1-based descending
        sorted_col = np.sort(X[:, col])                # ascending
        X[:, col] = sorted_col[rank_desc - 1]

        fval = f(X.sum(axis=1))
        fiter[niter] = fval
        niter += 1

        col = (col + 1) % d

        if niter > stalliter:
            fprev = fiter[niter - 1 - stalliter]
            converged = (fprev - fval) <= max(abs_tol, rel_tol * abs(fval))

    return _sort_and_pack(X, niter, fval, fiter, converged)


# ---------------------------------------------------------------------------
# BRA  —  Block Rearrangement Algorithm
# ---------------------------------------------------------------------------

def bra(
    X: NDArray,
    f: Callable[[NDArray], float] = _variance_of_sums,
    shuffle: bool = True,
    maxiter: int = 1_000,
    stalliter: int | None = None,
    abs_tol: float = 0.0,
    rel_tol: float = 0.0,
    f_target: float = -np.inf,
    seed: int | None = None,
) -> RearrangementResult:
    """Block Rearrangement Algorithm: at each step, randomly partition columns
    into two blocks and counter-monotonically rearrange the smaller block.
    """
    rng = np.random.default_rng(seed)
    X = X.copy()
    d = X.shape[1]
    if stalliter is None:
        stalliter = d
    if shuffle:
        X = shuffle_matrix(X, rng)

    fiter = np.empty(maxiter)
    fval = f(X.sum(axis=1))
    fiter[0] = fval
    niter = 1
    converged = False

    while fval > f_target and niter < maxiter and not converged:
        partition = rng.choice([True, False], size=d)
        X = rearrange_partition(X, partition)

        fval = f(X.sum(axis=1))
        fiter[niter] = fval
        niter += 1

        if niter > stalliter:
            fprev = fiter[niter - 1 - stalliter]
            converged = (fprev - fval) <= max(abs_tol, rel_tol * abs(fval))

    return _sort_and_pack(X, niter, fval, fiter, converged)


# ---------------------------------------------------------------------------
# BRAVE  —  BRA with Variance Equalization
# ---------------------------------------------------------------------------

def brave(
    X: NDArray,
    f: Callable[[NDArray], float] = _variance_of_sums,
    shuffle: bool = False,
    maxiter: int = 1_000,
    stalliter: int | None = None,
    abs_tol: float = 0.0,
    rel_tol: float = 0.0,
    f_target: float = -np.inf,
    version: int = 1,
    nochange_tol: float = 0.0,
    seed: int | None = None,
) -> RearrangementResult:
    """BRAVE algorithm: uses covariance-based partitions (equalvar) with
    fallback strategies when progress stalls.

    *version* controls the stall-escape strategy:
      1 — random block partition
      2 — random single column
      3 — sequential single column (RA-style)
      4 — permanently switch to RA
    """
    rng = np.random.default_rng(seed)
    X = X.copy()
    d = X.shape[1]
    if stalliter is None:
        stalliter = maxiter
    if shuffle:
        X = shuffle_matrix(X, rng)

    fiter = np.empty(maxiter)
    blocksizeiter = np.empty(maxiter, dtype=int)
    rand_flag = np.zeros(maxiter, dtype=bool)

    fval = f(X.sum(axis=1))
    fiter[0] = fval
    niter = 1
    converged = False
    partition_prev = np.zeros(d, dtype=bool)
    ra_col = 0
    ra_switch = False
    var_prev = np.inf

    while fval > f_target and niter < maxiter and not converged:
        partition = _equalvar(X, partition_prev, rng)
        blocksizeiter[niter] = min(partition.sum(), d - partition.sum())
        X = rearrange_partition(X, partition)
        fval = f(X.sum(axis=1))
        var_curr = float(np.var(X.sum(axis=1), ddof=1))
        fiter[niter] = fval
        niter += 1

        # Stall detection
        if (var_prev - var_curr) <= nochange_tol * var_curr and niter < maxiter:
            rand_flag[niter] = True
            if version == 1:
                partition = rng.choice([True, False], size=d)
                blocksizeiter[niter] = min(partition.sum(), d - partition.sum())
            elif version == 2:
                partition = np.zeros(d, dtype=bool)
                partition[rng.integers(d)] = True
                blocksizeiter[niter] = 1
            elif version == 3:
                partition = np.zeros(d, dtype=bool)
                partition[ra_col] = True
                ra_col = (ra_col + 1) % d
                blocksizeiter[niter] = 1
            elif version == 4:
                ra_switch = True
                partition = np.zeros(d, dtype=bool)

            if not ra_switch:
                X = rearrange_partition(X, partition)
                fval = f(X.sum(axis=1))
                fiter[niter] = fval
                niter += 1

        var_prev = float(np.var(X.sum(axis=1), ddof=1))
        partition_prev = partition

        if niter > stalliter:
            fprev = fiter[niter - 1 - stalliter]
            converged = ra_switch or (fprev - fval) <= max(abs_tol, rel_tol * abs(fval))

    # If version 4 triggered, finish with RA
    if ra_switch:
        ra_result = ra(
            X, f=f, shuffle=shuffle,
            maxiter=maxiter - niter + 1,
            stalliter=stalliter,
            abs_tol=abs_tol, rel_tol=rel_tol, f_target=f_target,
            seed=rng.integers(2**31),
        )
        n_ra = ra_result.niter
        # The RA history is spliced in starting at niter-1 so its first entry
        # overwrites the stale last pre-RA value (produced when ra_switch was
        # set). This is intentional — see the zero-partition branch above.
        fiter[niter - 1 : niter - 1 + n_ra] = ra_result.fiter
        rand_flag[niter - 1 : niter - 1 + n_ra] = True
        blocksizeiter[niter - 1 : niter - 1 + n_ra] = 1
        niter = niter - 1 + n_ra
        fval = ra_result.fval
        converged = ra_result.converged
        X = ra_result.X  # already sorted

        return RearrangementResult(
            X=X, S=ra_result.S, niter=niter, fval=fval,
            fiter=fiter[:niter], converged=converged,
            rand=rand_flag[:niter], blocksizeiter=blocksizeiter[:niter],
        )

    return _sort_and_pack(
        X, niter, fval, fiter, converged,
        rand=rand_flag[:niter],
        blocksizeiter=blocksizeiter[:niter],
    )
