"""Ranking metrics computed against the full catalogue (own implementation)."""
import numpy as np


def evaluate(model, users, seen, rel, n_items, k=10, chunk=1000):
    """Rank the full catalogue for each user, excluding items the user already has in `seen`.

    model.scores(users) -> dense array (len(users), n_items)
    seen: CSR users x items, every interaction known at fit time (masked from the ranking)
    rel:  CSR users x items, relevant (positive) target items
    Returns per-user arrays and the summary dict (metrics averaged over `users`).
    """
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    idcg_table = np.cumsum(discounts)
    prec, rec, ndcg, topks = [], [], [], []
    for s in range(0, len(users), chunk):
        u = users[s:s + chunk]
        sc = np.asarray(model.scores(u), dtype=np.float64)
        r, c = seen[u].nonzero()
        sc[r, c] = -np.inf
        part = np.argpartition(-sc, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(sc, part, axis=1)
        order = np.argsort(-vals, axis=1, kind="stable")
        topk = np.take_along_axis(part, order, axis=1)
        rel_d = rel[u].toarray() > 0
        hits = np.take_along_axis(rel_d, topk, axis=1)
        n_rel = rel_d.sum(1)
        h = hits.sum(1)
        prec.append(h / k)
        rec.append(h / n_rel)
        dcg = (hits * discounts).sum(1)
        ndcg.append(dcg / idcg_table[np.minimum(n_rel, k) - 1])
        topks.append(topk)
    prec, rec, ndcg = map(np.concatenate, (prec, rec, ndcg))
    topk = np.vstack(topks)
    summary = {
        f"Precision@{k}": float(prec.mean()),
        f"Recall@{k}": float(rec.mean()),
        f"NDCG@{k}": float(ndcg.mean()),
        f"Coverage@{k}": float(len(np.unique(topk)) / n_items),
    }
    return dict(prec=prec, rec=rec, ndcg=ndcg, topk=topk, summary=summary)


def bootstrap_ci(x, n_boot=1000, seed=42, alpha=0.05):
    """Percentile bootstrap CI of the mean, resampling users."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
