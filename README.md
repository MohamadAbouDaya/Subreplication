# subreplicating

Feasible sub-replicating portfolios for basket call options, built from
rearrangement-algorithm approximations of the worst-case dependence structure.

This package implements the method described in the paper *Feasible
sub-replicating portfolios for basket options via the block rearrangement
algorithm*: given marginal distributions and no dependence assumption, it
constructs a portfolio of vanilla options on individual assets whose price is
a model-free lower bound for the basket call.

## Install

```bash
pip install -e .
# optional extras
pip install -e ".[fast]"   # numba JIT for the vertex enumeration
pip install -e ".[dev]"    # pytest + ruff
```

Requires Python ≥ 3.10.

## Quick start

```python
import numpy as np
from subreplicating import lognormal_marginals, bra, sub_portfolio, refine_portfolio, call_price

# 3 equally-weighted log-normal marginals, T = 0.5
prices = lognormal_marginals(
    weights=np.array([100/3, 100/3, 100/3]),
    vols=np.array([0.355, 0.20, 0.25]),
    T=0.5,
    n=50_000,
)

# Approximate worst-case dependence with the Block Rearrangement Algorithm
r = bra(prices, maxiter=100_000, seed=0)

# Build the sub-replicating portfolio for strike K = 110
sr = sub_portfolio(r.X, kb=110.0, n_partition=100_000)
print(f"HV (phi-crossing strikes) = {sr.value:.5f}")

# Refine the strike values (local search on the hedging value, exact cash)
m, hv = refine_portfolio(r.X, sr, 110.0)
print(f"HV (refined)               = {hv:.5f}")
print(f"BV                        = {call_price(r.X.sum(axis=1), 110.0):.5f}")
```

`HV` is a valid lower bound for the basket call for *any* strikes (the cash
component is computed exactly by vertex enumeration), and `HV == BV` certifies
that the coupling `r.X` is optimal for the discretised marginals (weak duality,
Corollary 2.6 of the paper).  The price of the same portfolio under the exact
marginals is a valid lower bound for the continuous problem.

To certify the result for the exact marginals (Section 2.3 of the paper),
refine the strikes against the exact call prices and reassemble the coupling
with the exact marginals; the two numbers bracket the true lower bound:

```python
from subreplicating.exact import polish_strikes_exact, reassembled_basket_value

w, vols, T = np.array([100/3] * 3), np.array([0.355, 0.20, 0.25]), 0.5
m_ex, hv_ex = polish_strikes_exact(m, w, vols, T, 110.0, unpark_at=prices.max(axis=0))
bv_ex = reassembled_basket_value(r.X, w, vols, T, 110.0)
print(f"{hv_ex:.6f} <= lower bound of the basket call <= {bv_ex:.6f}")
```

Both functions also take a list of marginal objects in place of
`(weights, vols, T)`; `exact.py` provides log-normal, truncated normal-mixture
and affine-transformed marginals.

BRA and RA accept an alternative objective; `basket_call_objective(kb)` targets
`E[(sum - kb)^+]` directly and is non-increasing along every rearrangement step,
because each step is a convex-order decrease of the row sum:

```python
from subreplicating import basket_call_objective, ra
r2 = ra(prices, f=basket_call_objective(110.0), maxiter=100_000, stalliter=200, seed=0)
```

## Module map

| Module | Paper section | Contents |
|---|---|---|
| `marginals.py` | §5 (examples setup) | `lognormal_marginals` — generates marginal quantile samples |
| `rearrangement.py` | §2.2 | `ra`, `bra`, `brave` — rearrangement algorithms for worst-case dependence; `basket_call_objective` — K-aware objective |
| `subreplication.py` | §2.1–2.2 | `phi_calc`, `k_finder`, `sub_portfolio` (exact parity search, snap-to-K rule), `analytical_cash` (Proposition 2.3), `refine_portfolio` / `polish_strikes` (strike refinement: Nelder–Mead alternated with Powell), `port_value` |
| `exact.py` | §2.3 | `polish_strikes_exact` (lower bound under the exact marginals), `reassembled_basket_value` (upper bound, Proposition 2.11), marginal classes |
| `_fast.py` | §2.2 | optional Numba kernels used by `polish_strikes` |
| `examples/_common.py` | §2.2, Algorithm 1 | `run_pipeline` — restarts × algorithms × objectives, best-HV selection, strike refinement |
| `pricing.py` | §2 (payoffs) | `call_price`, `put_price` on simulated price vectors |
| `types.py` | — | Result dataclasses |

## Examples

```bash
python -m subreplicating.examples.three_asset            # discretised pipeline, three assets
python -m subreplicating.examples.six_asset              # discretised pipeline, six assets
python -m subreplicating.examples.exact_bracket 3        # Table 2 of the paper (all columns)
python -m subreplicating.examples.exact_bracket 6        # Table 3 of the paper (all columns)
python -m subreplicating.examples.case2_dominant_asset  # Table 4 of the paper (Case 2, one dominant asset)
```

The comparison with the optimal-transport algorithm of Neufeld and Xiang
(Tables 5 and 6) is in `comparison/neufeld_xiang/`; see the README there, in
particular the note on the substitute solvers.

The example scripts implement Algorithm 1 of the paper (RA and BRA, variance and
K-aware objectives, best-HV selection across restarts, strike refinement) and
parallelise the restarts across CPU cores via joblib.  The refinement is
applied before the result is compared with the two strike-free portfolios:
deep out of the money the unrefined phi-crossing strikes can be worth less than
those although, refined, they attain the basket value.  The stall window of the
rearrangement algorithms defaults to a few hundred iterations
(`examples._common.default_stalliter`); the older default of one iteration per
asset stopped BRA long before convergence.

## Development

```bash
pytest            # a few seconds
ruff check .
ruff format .
```

## License

MIT.
