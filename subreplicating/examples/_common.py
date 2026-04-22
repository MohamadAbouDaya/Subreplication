"""Shared helpers for the example scripts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from ..pricing import call_price
from ..subreplication import sub_portfolio

Algo = Callable[..., Any]
Objective = Callable[[np.ndarray], float] | None  # None = algo's default


def _one_restart(
    prices: np.ndarray,
    kb: float,
    algo: Algo,
    f: Objective,
    maxiter: int,
    seed: int,
    n_partition: int,
):
    kwargs = {"maxiter": maxiter, "stalliter": prices.shape[1], "seed": seed}
    if f is not None:
        kwargs["f"] = f
    r = algo(prices, **kwargs)
    sr = sub_portfolio(r.X, kb, n_partition=n_partition)
    return sr.value, r


def run_best_coupling(
    prices: np.ndarray,
    kb: float,
    algos: Sequence[Algo],
    restarts: int,
    maxiter: int,
    n_partition: int,
    objectives: Sequence[Objective] = (None,),
    n_jobs: int = -1,
) -> tuple[np.ndarray, float]:
    """Run each (algo, objective) pair `restarts` times in parallel, return the
    coupling that maximises the subreplicating portfolio value.

    `objectives=(None,)` uses each algorithm's default (variance). Pass a list
    such as `[None, basket_call_objective(kb)]` to also search with a K-aware
    objective — the best-HV output across all runs is returned.
    """
    tasks = [
        delayed(_one_restart)(prices, kb, algo, f, maxiter, seed, n_partition)
        for algo in algos
        for f in objectives
        for seed in range(restarts)
    ]
    results = Parallel(n_jobs=n_jobs)(tasks)

    _, best_r = max(results, key=lambda t: t[0])
    return best_r.X, best_r.fval


def print_report(prices_r: np.ndarray, kb: float, sr, fval: float) -> None:
    m = sr.m
    pb = prices_r.sum(axis=1)
    bv = call_price(pb, kb)
    ub = call_price(np.sort(prices_r, axis=0).sum(axis=1), kb)
    gap = (bv - sr.value) / bv * 100 if bv > 1e-8 else 0.0

    print(f"  fval = {fval:.8f}")
    print("  Strikes per asset:")
    for i, strikes in enumerate(m):
        print(f"    asset {i}: {np.round(strikes, 4)}")
    print(f"  Cash component:     {sr.cash:12.6f}")
    print(f"  Options component:  {sr.v:12.6f}")
    print(f"  Subrep. value (HV): {sr.value:12.6f}")
    print(f"  Basket call   (BV): {bv:12.6f}")
    print(f"  Upper bound:        {ub:12.6f}")
    print(f"  Gap (BV-HV)/BV:     {gap:11.2f}%")
    print()
