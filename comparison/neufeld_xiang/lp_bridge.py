"""Fallback LP solve with HiGHS for the Gurobi stand-in: min f'x s.t. A x <= b, x free."""
import sys

import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

d = loadmat(sys.argv[1])
f = np.asarray(d["f"], float).ravel()
A = csr_matrix(d["Aineq"])
b = np.asarray(d["bineq"], float).ravel()
res = None
for method in ("highs-ds", "highs-ipm"):
    res = linprog(f, A_ub=A, b_ub=b, bounds=[(None, None)] * len(f), method=method,
                  options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
    if res.status == 0:
        break
if res.status == 0:
    savemat(sys.argv[2], {"ok": 1.0, "x": res.x, "fval": res.fun, "lam": -res.ineqlin.marginals})
else:
    savemat(sys.argv[2], {"ok": 0.0})
