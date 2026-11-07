"""Part 1 baselines: MostPopular (variants) and item-based kNN.

Both models are fitted on a DataFrame with columns u, i, r, ts. The same data is
the user's history at scoring time (items in it are removed from the ranking by
the evaluator).
"""
import numpy as np

from data import SEC_PER_DAY, interaction_matrix


def _tie_break(df, n_items):
    """Deterministic secondary order: total train popularity, scaled into [0, 1)."""
    c = np.bincount(df["i"].to_numpy(), minlength=n_items).astype(np.float64)
    return c / (c.max() + 1.0)


class MostPopular:
    """Non-personalised popularity.

    positive_only: count only ratings >= pos_threshold (else all ratings).
    window_days:   count only ratings in the last `window_days` before the end of the
                   fit data (None = all-time).
    """

    def __init__(self, n_items, pos_threshold=4, positive_only=False, window_days=None):
        self.n_items, self.thr = n_items, pos_threshold
        self.positive_only, self.window_days = positive_only, window_days

    def fit(self, df, n_users):
        d = df
        if self.window_days is not None:
            d = d[d["ts"] > d["ts"].max() - self.window_days * SEC_PER_DAY]
        if self.positive_only:
            d = d[d["r"] >= self.thr]
        pop = np.bincount(d["i"].to_numpy(), minlength=self.n_items).astype(np.float64)
        # counts are integers, so the tie-break (< 1e-3) can never reorder distinct counts
        self.vec = pop + 1e-3 * _tie_break(df, self.n_items)
        return self

    def scores(self, users):
        return np.tile(self.vec, (len(users), 1))


class ItemKNN:
    """Item-based neighbourhood CF on positive implicit feedback.

    score(u, j) = sum over items i in the user's positive history of S_k[i, j],
    where S_k keeps, for every target item j, only its k most similar items.
    Negative similarities are clipped to 0.

    sim:
      cosine   - cosine between binary positive columns
      jaccard  - Jaccard between binary positive columns
      adjcos   - adjusted cosine: ratings centred by the user's mean rating
    """

    def __init__(self, n_items, pos_threshold=4, k=50, sim="cosine"):
        self.n_items, self.thr, self.k, self.sim = n_items, pos_threshold, k, sim

    def fit(self, df, n_users):
        self.X = interaction_matrix(df, n_users, self.n_items, True, self.thr)
        self.tb = _tie_break(df, self.n_items)
        self._similarity(df, n_users)
        self.set_k(self.k)
        return self

    def _similarity(self, df, n_users):
        import scipy.sparse as sp

        n = self.n_items
        if self.sim in ("cosine", "jaccard"):
            A = self.X
            co = (A.T @ A).toarray().astype(np.float32)
            cnt = np.asarray(A.sum(0)).ravel().astype(np.float32)
            if self.sim == "cosine":
                denom = np.sqrt(np.outer(cnt, cnt))
            else:
                denom = cnt[:, None] + cnt[None, :] - co
        elif self.sim == "adjcos":
            um = df.groupby("u")["r"].mean()
            centred = df["r"].to_numpy() - um.reindex(df["u"].to_numpy()).to_numpy()
            C = sp.csr_matrix((centred.astype(np.float32), (df["u"].to_numpy(), df["i"].to_numpy())),
                              shape=(n_users, n))
            co = (C.T @ C).toarray().astype(np.float32)
            nrm = np.sqrt(np.maximum(np.diag(co), 0))
            denom = np.outer(nrm, nrm)
        else:
            raise ValueError(self.sim)
        S = co / np.maximum(denom, 1e-12)
        S = np.maximum(S, 0)
        np.fill_diagonal(S, 0)
        self.S = S.astype(np.float32)

    def set_k(self, k):
        """Re-prune the stored similarity matrix to k neighbours per target item."""
        self.k = k
        S = self.S
        kk = int(min(k, S.shape[0] - 1))
        idx = np.argpartition(-S, kk - 1, axis=0)[:kk]
        mask = np.zeros(S.shape, dtype=bool)
        np.put_along_axis(mask, idx, True, axis=0)
        self.Sk = np.where(mask, S, 0).astype(np.float32)

    def scores(self, users):
        x = self.X[users].toarray()
        return (x @ self.Sk).astype(np.float64) + 1e-9 * self.tb
