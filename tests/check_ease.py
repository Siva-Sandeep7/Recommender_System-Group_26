"""Checks EASE: (1) the closed form equals explicit per-item ridge regressions with diag(B)=0,
(2) the diagonal is zero, (3) it learns real structure (beats popularity on clustered data).

    python tests/check_ease.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data  # noqa: E402
import ease  # noqa: E402
import metrics  # noqa: E402
import models  # noqa: E402

rng = np.random.default_rng(5)
n_u, n_i, lam = 60, 25, 3.0
M = rng.random((n_u, n_i)) < 0.3
rows, cols = np.nonzero(M)
df = pd.DataFrame({"u": rows, "i": cols, "r": 5, "ts": 0})
m = ease.EASE(n_i, 4, lam).fit(df, n_u)
X = M.astype(float)

# 1. one ridge regression per item j, using only the other items as predictors
err = 0.0
for j in range(n_i):
    others = [i for i in range(n_i) if i != j]
    A = X[:, others]
    w = np.linalg.solve(A.T @ A + lam * np.eye(n_i - 1), A.T @ X[:, j])
    err = max(err, np.abs(w - m.B[others, j]).max())
print("closed form vs explicit ridge regressions: max |diff| = %.2e" % err)
assert err < 1e-4
print("1. closed form equals the constrained ridge regressions. OK")

assert np.all(np.diag(m.B) == 0)
print("2. diag(B) = 0 (no item predicts itself). OK")

# 3. learning test on clustered data
n_u, n_i, n_cl = 300, 200, 4
cl_u, cl_i = rng.integers(0, n_cl, n_u), rng.integers(0, n_cl, n_i)
rr = []
for u in range(n_u):
    items = np.nonzero(rng.random(n_i) < np.where(cl_i == cl_u[u], 0.35, 0.03))[0]
    rng.shuffle(items)
    rr += [(u, i, 5, t) for t, i in enumerate(items)]
full = pd.DataFrame(rr, columns=["user", "item", "rating", "ts"])
sp_ = data.temporal_split(full, mode="per_user")
seen = data.interaction_matrix(sp_.train, sp_.n_users, sp_.n_items)
rel = data.interaction_matrix(sp_.test, sp_.n_users, sp_.n_items, True, 4)
users = np.where(np.asarray(rel.sum(1)).ravel() > 0)[0]
e = ease.EASE(sp_.n_items, 4, 50.0).fit(sp_.train, sp_.n_users)
a = metrics.evaluate(e, users, seen, rel, sp_.n_items, 10)["summary"]["NDCG@10"]
p = metrics.evaluate(models.MostPopular(sp_.n_items).fit(sp_.train, sp_.n_users), users, seen, rel, sp_.n_items, 10)["summary"]["NDCG@10"]
print("EASE NDCG@10 %.3f   MostPop %.3f   (cluster oracle is about 0.13)" % (a, p))
assert a > 0.10 and a > 3 * p
print("3. EASE recovers the cluster structure. OK")
