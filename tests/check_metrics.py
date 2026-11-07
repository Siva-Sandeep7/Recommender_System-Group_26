"""Checks the self-written NDCG@10 against scikit-learn's reference on a random sample.

    python tests/check_metrics.py
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import ndcg_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import metrics  # noqa: E402


class FixedScores:
    def __init__(self, s):
        self.s = s

    def scores(self, users):
        return self.s[users].copy()


rng = np.random.default_rng(42)
n_u, n_i, k = 200, 300, 10
scores = rng.random((n_u, n_i))
rel = sp.csr_matrix((rng.random((n_u, n_i)) < 0.03).astype(np.float64))
keep = np.where(np.asarray(rel.sum(1)).ravel() > 0)[0]
seen = sp.csr_matrix((n_u, n_i), dtype=np.float32)
res = metrics.evaluate(FixedScores(scores), keep, seen, rel, n_i, k)
ref = np.mean([ndcg_score(rel[u].toarray(), scores[u:u + 1], k=k) for u in keep])
print("own NDCG@10 = %.6f   sklearn = %.6f   |diff| = %.2e" % (res["summary"]["NDCG@10"], ref,
                                                              abs(res["summary"]["NDCG@10"] - ref)))
assert abs(res["summary"]["NDCG@10"] - ref) < 1e-9
# precision / recall by direct definition
top = np.argsort(-scores, axis=1)[:, :k]
P = np.mean([rel[u].toarray().ravel()[top[u]].sum() / k for u in keep])
R = np.mean([rel[u].toarray().ravel()[top[u]].sum() / float(rel[u].toarray().sum(dtype=np.float64)) for u in keep])
assert abs(P - res["summary"]["Precision@10"]) < 1e-9 and abs(R - res["summary"]["Recall@10"]) < 1e-9
print("Precision@10 and Recall@10 also match the direct definition. OK")
