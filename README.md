# subreplicating

Feasible sub-replicating portfolios for basket call options, built from
rearrangement-algorithm approximations of the worst-case dependence structure.

This package implements the method described in the accompanying paper
(`main.tex`): given marginal distributions and no dependence assumption, it
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
from subreplicating import lognormal_marginals, bra, sub_portfolio, call_price

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
print(f"HV = {sr.value:.5f}")
print(f"BV = {call_price(r.X.sum(axis=1), 110.0):.5f}")
```

## Module map

| Module | Paper section | Contents |
|---|---|---|
| `marginals.py` | §3 (examples setup) | `lognormal_marginals` — generates marginal quantile samples |
| `rearrangement.py` | §2.3 | `ra`, `bra`, `brave`, `basket_call_objective` — rearrangement algorithms for worst-case dependence |
| `subreplication.py` | §2.1–2.2 | `phi_calc`, `k_finder`, `sub_portfolio`, `port_value`, `polish_strikes`, `analytical_cash`, `monte_carlo_cash` |
| `pricing.py` | §2 (payoffs) | `call_price`, `put_price` on simulated price vectors |
| `types.py` | — | Result dataclasses |

## Examples

```bash
python -m subreplicating.examples.three_asset        # baseline 3-asset run
python -m subreplicating.examples.three_asset_best   # tightest-fit: BRA+RA × {variance, K-aware} + Nelder-Mead polish
python -m subreplicating.examples.six_asset          # 6-asset basket
```

All scripts parallelise BRA/RA restarts across CPU cores via joblib.

## Reproducing the paper

The two numerical tables in §4 of `main.tex` are produced by the example scripts above.
Both use $t = 0.5$, equal weights, $N = 50{,}000$ marginal discretisation steps, and
$n_{\text{partition}} = 100{,}000$ grid points for $\phi_i$.

**Table 1 — three-asset basket** (vols 35.5%, 20%, 25%; strikes 100, 105, 110, 115, 120):

```bash
python -m subreplicating.examples.three_asset_best
```

Runs BRA and RA at 250 restarts each with both the variance proxy and the K-aware
objective $E[(\Sigma s - K)^+]$, selects the coupling by $HV$, then applies the
Nelder–Mead strike polish. Produces the asset strikes, cash, $HV$, and $BV$
columns of Table 1.

**Table 2 — six-asset basket** (vols 91.5%, 75.5%, 82.5%, 35.5%, 20%, 15%;
strikes 100, 110, 120, 130, 140):

```bash
python -m subreplicating.examples.six_asset
```

Runs BRA at 20 restarts (variance proxy) and takes zero-crossings of $\phi_i$ to
construct the portfolio. Produces the columns of Table 2.

### Reproducibility notes

- BRA/RA are randomised. The scripts do not fix a global seed, so individual
  numbers will drift at the fifth decimal across runs. The qualitative claim of
  §4.1 — the polished $HV$ matches $BV$ to four–five decimals at every strike —
  reproduces run-to-run.
- Runtime scales with `BRA_RESTARTS` × `BRA_MAXITER`. With the defaults, the
  3-asset best-fit script takes a few minutes on a modern laptop; the 6-asset
  script takes longer per strike because each BRA step is more expensive.
- Install the `fast` extra (`pip install -e ".[fast]"`) to JIT the vertex
  enumeration inside `sub_portfolio` — this is the dominant cost for the
  6-asset run.
- To change the setup, edit `WEIGHTS`, `IMPLIED_VOLS`, `STRIKES`, `N`, and
  `BRA_RESTARTS` at the top of the example files.

## Development

```bash
pytest            # ~0.5s
ruff check .
ruff format .
```

## License

MIT.
