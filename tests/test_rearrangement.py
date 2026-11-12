import numpy as np
import pytest

from subreplicating.rearrangement import (
    _greedy_partition,
    _rank_descending,
    basket_call_objective,
    bra,
    ra,
    rearrange,
    rearrange_partition,
    shuffle_matrix,
)


def _simple_matrix():
    return np.array(
        [
            [1.0, 5.0, 9.0],
            [2.0, 6.0, 8.0],
            [3.0, 7.0, 7.0],
            [4.0, 8.0, 6.0],
        ]
    )


def test_rank_descending_no_ties():
    v = np.array([3.0, 1.0, 4.0, 1.5, 9.0])
    ranks = _rank_descending(v)
    # 9.0 -> 1, 4.0 -> 2, 3.0 -> 3, 1.5 -> 4, 1.0 -> 5
    assert ranks.tolist() == [3, 5, 2, 4, 1]


def test_rank_descending_ties_break_by_position():
    v = np.array([2.0, 2.0, 1.0])
    ranks = _rank_descending(v)
    # Mergesort is stable; the first 2.0 ranks above the second.
    assert ranks.tolist() == [1, 2, 3]


def test_rearrange_counter_monotone():
    # The defining property: row ranks of block1 sums (after permutation)
    # must be the reverse of row ranks of block2 sums.
    rng = np.random.default_rng(0)
    b1 = rng.normal(size=(8, 3))
    b2 = rng.normal(size=(8, 2))
    out = rearrange(b1, b2)
    s1 = out.sum(axis=1)
    s2 = b2.sum(axis=1)
    # Largest s2 should pair with smallest s1, second-largest with second-smallest, ...
    order2_desc = np.argsort(-s2, kind="mergesort")
    order1_asc = np.argsort(s1, kind="mergesort")
    assert order2_desc.tolist() == order1_asc.tolist()


def test_rearrange_partition_preserves_columns():
    X = _simple_matrix()
    part = np.array([True, False, False])
    out = rearrange_partition(X, part)
    # Each column is just a permutation of the original column.
    for j in range(X.shape[1]):
        assert sorted(out[:, j].tolist()) == sorted(X[:, j].tolist())


def test_shuffle_matrix_preserves_columns():
    X = _simple_matrix()
    rng = np.random.default_rng(42)
    out = shuffle_matrix(X, rng)
    for j in range(X.shape[1]):
        assert sorted(out[:, j].tolist()) == sorted(X[:, j].tolist())


def test_greedy_partition_balances():
    vals = np.array([8.0, 7.0, 6.0, 5.0, 4.0])
    diff, _ = _greedy_partition(vals)
    # Optimal balanced partition is (8,5,4)=17 vs (7,6)=13; greedy reaches diff <= 4.
    assert diff <= 4.0


@pytest.mark.parametrize("algo", [ra, bra])
@pytest.mark.parametrize("objective", [None, "basket_call"])
def test_algorithms_non_increasing_objective(algo, objective):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 4)) + 5.0
    kwargs = {"maxiter": 200, "stalliter": 200, "seed": 7}
    if objective == "basket_call":
        # Each counter-monotonic block step is the antimonotone coupling of the
        # two block sums, hence a convex-order decrease of the row sum: any
        # convex objective such as E[(S - K)^+] is non-increasing along the run.
        kwargs["f"] = basket_call_objective(20.0)
    res = algo(X, **kwargs)
    diffs = np.diff(res.fiter)
    assert (diffs <= 1e-9).all(), "objective increased between iterations"
    # Output preserves each column's multiset.
    for j in range(X.shape[1]):
        assert sorted(res.X[:, j].tolist()) == sorted(X[:, j].tolist())


def test_ra_seed_is_reproducible():
    X = np.random.default_rng(0).normal(size=(40, 3))
    a = ra(X, maxiter=100, seed=123)
    b = ra(X, maxiter=100, seed=123)
    assert np.allclose(a.X, b.X)
    assert a.fval == b.fval
