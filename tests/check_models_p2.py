"""Checks UserKNN against a slow, obviously-correct loop implementation, and (if the `implicit`
package is installed) that the ALS wrapper runs and beats popularity on clustered data.

    python tests/check_models_p2.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data  # noqa: E402
import metrics  # noqa: E402
import models  # noqa: E402
import models_p2  # noqa: E402

rng = np.random.default_rng(3)
n_u, n_i = 40, 30
M = rng.random((n_u, n_i)) < 0.25
M[:, 0] = True            # item 0 is a blockbuster
M[5] = False              # a user with no positives at all
rows, cols = np.nonzero(M)
df = pd.DataFrame({"u": rows, "i": cols, "r": 5, "ts": 0})


def naive(M, k, sim, idf, shrink):
    U, I = M.shape
    X = M.astype(float)
    pop = X.sum(0)
    w = np.log((U + 1.0) / (pop + 1.0)) if idf else np.ones(I)
    S = np.zeros((U, U))
    for a in range(U):
        for b in range(U):
            if a == b:
                continue
            co = (X[a] * X[b]).sum()
            if sim == "cosine":
                num = (w * X[a] * X[b]).sum()
                den = np.sqrt((w * X[a]).sum() * (w * X[b]).sum())
            else:
                num, den = co, X[a].sum() + X[b].sum() - co
            s = num / den if den > 0 else 0.0
            if shrink > 0:
                s *= min(co, shrink) / shrink
            S[a, b] = s
    out = np.zeros((U, I))
    for a in range(U):
        nb = np.argsort(-S[a], kind="stable")[:k]
        out[a] = S[a, nb] @ X[nb]
    return S, out


for sim, idf, shrink in [("cosine", False, 0), ("cosine", True, 0), ("cosine", True, 3), ("jaccard", False, 0), ("jaccard", False, 2)]:
    # (a) with all other users as neighbours there is no tie problem: scores must be identical
    S_all, want = naive(M, n_u - 1, sim, idf, shrink)
    m = models_p2.UserKNN(n_i, 4, k=n_u - 1, sim=sim, idf=idf, shrink=shrink).fit(df, n_u)
    got = m.scores(np.arange(n_u))
    assert np.abs(got - want).max() < 1e-5, (sim, idf, shrink, "all neighbours")
    # (b) with k < all, equally similar users may be picked in a different order, so check that the
    #     similarities of the chosen neighbours are exactly the k largest ones for every user
    for k in (3, 10):
        S_ref, _ = naive(M, k, sim, idf, shrink)
        mk = models_p2.UserKNN(n_i, 4, k=k, sim=sim, idf=idf, shrink=shrink).fit(df, n_u)
        for a in range(n_u):
            want_vals = np.sort(S_ref[a])[::-1][:k]
            got_vals = np.sort(mk.Sk[a].toarray().ravel())[::-1][:k]
            assert np.allclose(want_vals, got_vals, atol=1e-5), (sim, idf, shrink, k, a)
    print("UserKNN %-8s idf=%-5s shrink=%d  exact with all neighbours; top-k similarities correct" % (sim, idf, shrink))
print("UserKNN matches the loop implementation. OK")

# --- optional: ALS wrapper ----------------------------------------------------------------
try:
    import implicit  # noqa: F401
except ImportError:
    print("implicit is not installed here: skipping the ALS check (run it on the machine that has it).")
    sys.exit(0)

n_u, n_i, n_cl = 300, 200, 4
cl_u, cl_i = rng.integers(0, n_cl, n_u), rng.integers(0, n_cl, n_i)
rows = []
for u in range(n_u):
    items = np.nonzero(rng.random(n_i) < np.where(cl_i == cl_u[u], 0.35, 0.03))[0]
    rng.shuffle(items)
    rows += [(u, i, 5, t) for t, i in enumerate(items)]
full = pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])
sp_ = data.temporal_split(full, mode="per_user")
seen = data.interaction_matrix(sp_.train, sp_.n_users, sp_.n_items)
rel = data.interaction_matrix(sp_.test, sp_.n_users, sp_.n_items, True, 4)
users = np.where(np.asarray(rel.sum(1)).ravel() > 0)[0]
als = models_p2.ImplicitALS(sp_.n_items, 4, factors=8, reg=0.1, alpha=10, iterations=10).fit(sp_.train, sp_.n_users)
a = metrics.evaluate(als, users, seen, rel, sp_.n_items, 10)["summary"]["NDCG@10"]
p = metrics.evaluate(models.MostPopular(sp_.n_items).fit(sp_.train, sp_.n_users), users, seen, rel, sp_.n_items, 10)["summary"]["NDCG@10"]
print("ALS NDCG@10 %.3f   MostPop %.3f   (cluster oracle is about 0.13)" % (a, p))
assert a > 0.10 and a > 3 * p
print("ImplicitALS wrapper works. OK")
