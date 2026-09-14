"""Shared helpers for the example scripts: Algorithm 1 of the paper.

Pipeline for one basket strike ``kb``:

  1. run each rearrangement algorithm (RA / BRA) with each objective
     (variance proxy, K-aware basket-call objective) for ``restarts`` random
     initialisations and keep the coupling with the highest raw hedging value
     HV = v + cash (exact cash, Proposition 2.1);
  2. compute the phi-crossing strikes for that coupling (``sub_portfolio``,
     which also performs the exact parity search and the snap-to-kb rule);
  3. polish the strike values with Nelder-Mead (``polish_strikes``);
  4. report HV (raw and polished), the BRA-approximated basket value BV and the
     co-monotone upper bound.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from ..pricing import call_price
from ..subreplication import polish_strikes, port_value, sub_portfolio

Algo = Callable[..., Any]
Objective = Callable[[np.ndarray], float] | None  # None = algo's default (variance)


def default_stalliter(n_assets: int) -> int:
    """Stall window for RA / BRA: stop after this many consecutive iterations
    without a strict improvement of the objective.

    The old default (= number of assets) stopped BRA after a handful of random
    block partitions -- for three assets a quarter of the random partitions are
    no-ops, so runs typically ended after 6-14 iterations with a variance an
    order of magnitude above the converged value.  A window of a few hundred
    iterations lets every restart converge before the best-of-restarts
    selection; the extra cost is negligible next to the restart budget.
    """
    return max(200, 50 * n_assets)


def _one_restart(
    prices: np.ndarray,
    kb: float,
    algo: Algo,
    f: Objective,
    maxiter: int,
    seed: int,
    n_partition: int,
    stalliter: int,
):
    kwargs: dict[str, Any] = {"maxiter": maxiter, "stalliter": stalliter, "seed": seed}
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
    stalliter: int | None = None,
    n_jobs: int = -1,
) -> tuple[np.ndarray, float]:
    """Run each (algo, objective) pair ``restarts`` times in parallel and return
    the coupling that maximises the (raw) subreplicating portfolio value.

    ``objectives=(None,)`` uses each algorithm's default (variance).  Pass e.g.
    ``[None, basket_call_objective(kb)]`` to also search with the K-aware
    objective; the best-HV output across all runs is returned.
    """
    if stalliter is None:
        stalliter = default_stalliter(prices.shape[1])
    tasks = [
        delayed(_one_restart)(prices, kb, algo, f, maxiter, seed, n_partition, stalliter)
        for algo in algos
        for f in objectives
        for seed in range(restarts)
    ]
    results = Parallel(n_jobs=n_jobs)(tasks)

    _, best_r = max(results, key=lambda t: t[0])
    return best_r.X, best_r.fval


@dataclass
class PipelineResult:
    """Output of :func:`run_pipeline` for one basket strike."""

    kb: float
    strikes_raw: list[np.ndarray]   # phi-crossing strikes (after parity search / snap)
    hv_raw: float                   # HV at the raw strikes
    strikes: list[np.ndarray]       # polished strikes
    cash: float                     # exact cash component at the polished strikes
    v: float                        # vanilla-option component at the polished strikes
    hv: float                       # polished hedging value = cash + v
    bv: float                       # basket call value under the selected coupling
    ub: float                       # co-monotone (upper) bound, for reference


def run_pipeline(
    prices: np.ndarray,
    kb: float,
    algos: Sequence[Algo],
    restarts: int,
    maxiter: int,
    n_partition: int,
    objectives: Sequence[Objective] = (None,),
    stalliter: int | None = None,
    polish: bool = True,
    n_jobs: int = -1,
) -> PipelineResult:
    """Algorithm 1 for a single strike; see the module docstring."""
    prices_r, _ = run_best_coupling(
        prices, kb, algos, restarts, maxiter, n_partition,
        objectives=objectives, stalliter=stalliter, n_jobs=n_jobs,
    )
    sr = sub_portfolio(prices_r, kb, n_partition=n_partition)
    if polish:
        m, _ = polish_strikes(prices_r, sr.m, kb)
    else:
        m = sr.m
    pv = port_value(prices_r, m, kb)
    return PipelineResult(
        kb=kb,
        strikes_raw=sr.m,
        hv_raw=sr.value,
        strikes=m,
        cash=pv.cash,
        v=pv.v,
        hv=pv.value,
        bv=call_price(prices_r.sum(axis=1), kb),
        ub=call_price(np.sort(prices_r, axis=0).sum(axis=1), kb),
    )


def print_report(res: PipelineResult) -> None:
    gap = (res.bv - res.hv) / res.bv * 100 if res.bv > 1e-8 else 0.0
    print(f"  Basket strike K = {res.kb:g}")
    print("  Strikes per asset (raw -> polished):")
    for i, (raw, pol) in enumerate(zip(res.strikes_raw, res.strikes, strict=True)):
        print(f"    asset {i}: {np.round(raw, 4).tolist()} -> {np.round(pol, 4).tolist()}")
    print(f"  Cash component:          {res.cash:12.6f}")
    print(f"  Options component:       {res.v:12.6f}")
    print(f"  HV raw (phi strikes):    {res.hv_raw:12.6f}")
    print(f"  HV polished:             {res.hv:12.6f}")
    print(f"  Basket call value (BV):  {res.bv:12.6f}")
    print(f"  Co-monotone upper bound: {res.ub:12.6f}")
    print(f"  Gap (BV-HV)/BV:          {gap:11.4f}%")
    print()
