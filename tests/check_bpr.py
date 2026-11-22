"""Verifies the self-implemented BPR.

    python tests/check_bpr.py

1. The analytic gradients equal central finite differences of the objective.
2. A batch of size 1 follows exactly the textbook update rule on one hand-made triple.
3. Sampled negatives are never items the user has touched.
4. Training raises the objective and the model clearly beats random ranking on held-out data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bpr  # noqa: E402
import data  # noqa: E402
import metrics  # noqa: E402
import models  # noqa: E402

rng = np.random.default_rng(0)

# 1 --- finite-difference gradient check -----------------------------------------------------
k, reg = 5, 0.3
p, qi, qj = rng.normal(size=k), rng.normal(size=k), rng.normal(size=k)
bi, bj = rng.normal(), rng.normal()
gp, gqi, gqj, gbi, gbj, _ = bpr.triple_grads(p[None], qi[None], qj[None], np.array([bi]), np.array([bj]), reg)


def num_grad(f, x, eps=1e-6):
    x = np.array(x, dtype=float)
    out = np.zeros_like(x)
    for t in range(x.size):
        d = np.zeros_like(x)
        d.flat[t] = eps
        out.flat[t] = (f(x + d) - f(x - d)) / (2 * eps)
    return out


obj = bpr.triple_objective
checks = {
    "p_u": (gp[0], num_grad(lambda v: obj(v, qi, qj, bi, bj, reg), p)),
    "q_i": (gqi[0], num_grad(lambda v: obj(p, v, qj, bi, bj, reg), qi)),
    "q_j": (gqj[0], num_grad(lambda v: obj(p, qi, v, bi, bj, reg), qj)),
    "b_i": (gbi[0], num_grad(lambda v: obj(p, qi, qj, v[0], bj, reg), [bi])[0]),
    "b_j": (gbj[0], num_grad(lambda v: obj(p, qi, qj, bi, v[0], reg), [bj])[0]),
}
for name, (a, n) in checks.items():
    err = np.max(np.abs(np.asarray(a) - np.asarray(n)))
    print("gradient wrt %-3s  max |analytic - numeric| = %.2e" % (name, err))
    assert err < 1e-7, name
print("1. gradients match finite differences. OK")

# 2 --- one SGD step with batch_size = 1 equals the textbook rule ---------------------------
df = pd.DataFrame({"u": [0, 0, 1], "i": [0, 1, 2], "r": [5, 5, 5], "ts": [1, 2, 3]})
m = bpr.BPR(n_items=4, k=3, lr=0.1, reg=0.05, epochs=1, batch_size=1, seed=7)
m.fit(df.iloc[:1], n_users=2)  # a single positive: (u=0, i=0)
r = np.random.default_rng(7)
P0 = r.normal(0, 0.1, (2, 3))
Q0 = r.normal(0, 0.1, (4, 3))
assert np.allclose(m.P[1], P0[1])  # untouched user keeps its initial vector
# the sampled negative is the only item whose item-bias moved besides item 0
j = [t for t in range(1, 4) if m.b[t] != 0.0]
assert len(j) == 1, "exactly one negative must have been updated"
j = j[0]
x = P0[0] @ (Q0[0] - Q0[j])
g = 1 / (1 + np.exp(x))
assert np.allclose(m.b[0], 0.1 * g) and np.allclose(m.b[j], -0.1 * g)
assert np.allclose(m.P[0], P0[0] + 0.1 * (g * (Q0[0] - Q0[j]) - 0.05 * P0[0]))
assert np.allclose(m.Q[0], Q0[0] + 0.1 * (g * P0[0] - 0.05 * Q0[0]))
assert np.allclose(m.Q[j], Q0[j] + 0.1 * (-g * P0[0] - 0.05 * Q0[j]))
print("2. batch_size=1 reproduces the hand-computed update. OK")

# 3 --- negatives are never touched items -----------------------------------------------------
n_u, n_i = 50, 40
touched = rng.random((n_u, n_i)) < 0.5
rows, cols = np.nonzero(touched)
d3 = pd.DataFrame({"u": rows, "i": cols, "r": 5, "ts": 0})
m3 = bpr.BPR(n_items=n_i, k=4, epochs=1)
m3._touched = touched
for alpha in (0.0, 0.75):
    m3.neg_alpha = alpha
    if alpha:
        m3._cdf = np.cumsum(np.ones(n_i) ** alpha / n_i)
    u = rng.integers(0, n_u, 5000)
    j = m3._sample_negatives(u, rng)
    assert not touched[u, j].any()
print("3. sampled negatives are never items the user has touched. OK")

# 4 --- it learns ---------------------------------------------------------------------------
n_u, n_i, n_cl = 300, 200, 4
cl_u, cl_i = rng.integers(0, n_cl, n_u), rng.integers(0, n_cl, n_i)
rows = []
for u in range(n_u):
    p_like = np.where(cl_i == cl_u[u], 0.35, 0.03)
    items = np.nonzero(rng.random(n_i) < p_like)[0]
    rng.shuffle(items)  # random time order, so train and test items are exchangeable
    for t, i in enumerate(items):
        rows.append((u, i, 5, t))
full = pd.DataFrame(rows, columns=["u", "i", "r", "ts"])
split = data.temporal_split(full.rename(columns={"u": "user", "i": "item", "r": "rating"}).assign(ts=lambda d: d["ts"]),
                            mode="per_user")
train, test = split.train, split.test
seen = data.interaction_matrix(train, split.n_users, split.n_items)
rel = data.interaction_matrix(test, split.n_users, split.n_items, True, 4)
users = np.where(np.asarray(rel.sum(1)).ravel() > 0)[0]
mb = bpr.BPR(split.n_items, k=8, lr=0.05, reg=0.01, epochs=30, batch_size=64, seed=1).fit(train, split.n_users)
h = mb.history_
print("   mean ln sigma(x) per epoch: first %.3f -> last %.3f" % (h[0], h[-1]))
assert h[-1] > h[0] + 0.05
res = metrics.evaluate(mb, users, seen, rel, split.n_items, 10)["summary"]
pop = metrics.evaluate(models.MostPopular(split.n_items).fit(train, split.n_users), users, seen, rel, split.n_items, 10)["summary"]
print("   NDCG@10  BPR %.3f   MostPop %.3f" % (res["NDCG@10"], pop["NDCG@10"]))
# an oracle that knows each user's cluster scores about 0.13 on this data, popularity about 0.035
assert res["NDCG@10"] > 0.10 and res["NDCG@10"] > 3 * pop["NDCG@10"]
print("4. training raises the objective and recovers the cluster structure (>> popularity). OK")
