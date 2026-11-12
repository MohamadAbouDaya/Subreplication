# Comparison with Neufeld and Xiang (arXiv:2203.01633)

Scripts behind Sections 5.2 and 5.3 of the paper: the multi-marginal
optimal-transport algorithm of Neufeld and Xiang applied to the basket-call
lower bound, and our construction applied to marginals of their Experiment 3.

## Important caveat: substitute solvers

Their implementation calls **Gurobi**. These runs did not use Gurobi.
`gurobi.m` in this folder is a stand-in with the same calling convention that
solves

* linear programmes with MATLAB `linprog` (dual simplex); the answer is
  accepted only if it passes the LP optimality conditions, otherwise the same
  LP is solved with HiGHS through `lp_bridge.py` (needs `python` with SciPy on
  the PATH);
* mixed-integer programmes with MATLAB `intlinprog`, which returns one
  solution per call where Gurobi returns a pool of up to 100.

Their algorithm files are used unmodified, but iteration counts and running
times are those of the stand-in, and the runs with equally spaced knots and
the finest grids did not converge within their time limit. To repeat the
comparison with Gurobi, delete or rename `gurobi.m` so that the real interface
is found.

Every lower bound is checked independently of MATLAB by `verify.py`, which
re-prices the dual solution and recomputes the minimum of payoff minus
portfolio by exact vertex enumeration (three assets) and with HiGHS.

## Running

1. Clone https://github.com/qikunxiang/MultiMarginalOptimalTransport into
   `../repo` relative to these scripts (i.e. `comparison/repo`), or edit the
   `repo = ...` line at the top of the drivers.
2. From this folder, in MATLAB (R2021a was used; Optimization and Statistics
   toolboxes):

   ```matlab
   % equal-probability knots (their Experiment 3 placement)
   run_basket('ex3', [0.355 0.20 0.25], [100 105 110 115 120], [16 32 64 128 256], 1e-6, 1e6, 10, 1200)
   run_basket('ex6', [0.915 0.755 0.825 0.355 0.20 0.15], [100 110 120 130 140], [4 8 16 32 64], 1e-6, 1e6, 10, 1200)
   % equally spaced knots, marginals truncated at the 1e-8 quantiles
   run_basket_uniform('v3_110', [0.355 0.20 0.25], 110, 64, 1e-8, 1e-6, 1e6, 10, 2400)
   % strikes of a preliminary refinement of our portfolios added to 4 and 16
   % equal-probability knots (results/fk3.json, fk6.json; see step 5)
   knots_final3
   knots_final6
   % one term of their Experiment 3 cost on their first three marginals
   export_exp3_marginals(10)
   run_exp3_term(3, [0 1 2], [4 8 16 32 64], 1e-6, 1e6, 10, 900)
   ```

   `exp3_cost.json` holds the direction vectors and intercepts of their
   Experiment 3 cost (same seed as their `CPWAMM_prepare.m`).
3. Check the output: `python verify.py ex3` (and likewise for the other tags).
4. Our side on their marginals, from this folder with the package importable:

   ```bash
   python exp3_term_ours.py 3 4 0 1 2     # |<s,x> - t| on three marginals
   python exp3_ours.py 6 2 5 6 8          # plain basket call on six marginals
   ```

5. The augmented-grid experiment of Section 5.2 (no MATLAB needed):

   ```bash
   python augmented_grid_lp.py 3          # strikes of results/final3.json + 4 and 16 knots
   python augmented_grid_lp.py 6
   python augmented_grid_lp.py --check fk3   # validation against their code, see below
   python augmented_grid_lp.py --check fk6
   ```

   On a grid of knots their lower bound is, up to the tolerance of their
   programme, the value of the best sub-replicating portfolio with strikes at
   the knots, and that value is a finite linear programme (one constraint per
   vertex).  `augmented_grid_lp.py` solves it to optimality by constraint
   generation and compares the optimum with the value of our portfolio
   (`results/grid_lp_3.json`, `grid_lp_6.json`).  With `--check` it solves the
   same programme on the knots stored in `results/fk3.json` / `fk6.json`, the
   runs of their code with the strikes of a preliminary refinement, and prints
   the optimum next to the bound their code returned: on these twenty grids
   the two differ by at most 3e-7 (`results/grid_lp_check.txt`).

Our own brackets for the log-normal examples come from
`python -m subreplicating.examples.exact_bracket 3` (and `6`); their output is
`results/final3.json` and `final6.json` (Tables 2 and 3 of the paper).

## Files

| File | Purpose |
|---|---|
| `gurobi.m`, `lp_bridge.py` | stand-in for the Gurobi MATLAB interface |
| `ProbMeas1DLogNormal.m` | truncated log-normal marginal for their framework |
| `run_basket.m` | their solver, equal-probability knots |
| `run_basket_uniform.m` | their solver, equally spaced knots |
| `run_basket_knots.m`, `knots_final*.m` | their solver with strikes of ours added as knots |
| `augmented_grid_lp.py` | best portfolio with strikes on an augmented grid, as a linear programme |
| `run_exp3_term.m`, `export_exp3_marginals.m` | their solver on one term of their Experiment 3 cost |
| `exp3_term_ours.py`, `exp3_ours.py` | our construction on their marginals |
| `verify.py` | independent check of their dual solutions |
| `results/` | JSON output and text logs of the runs reported in the paper |
