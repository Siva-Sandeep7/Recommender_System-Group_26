"""Part 2 models that are NOT self-implemented in the sense of the project requirement
(the self-implemented algorithm is bpr.py):

  UserKNN      user-based neighbourhood CF on positive implicit feedback (Lecture 1.2, slides 5-11).
               Written with NumPy/SciPy; no recommender library.
  ImplicitALS  weighted matrix factorisation fitted with alternating least squares
               (Lecture 2.1, slides 9-10), taken from the `implicit` library.

Both follow the project's model interface: fit(df, n_users) and scores(users).
"""
import numpy as np

from data import interaction_matrix


class UserKNN:
    """score(u, j) = sum over the k nearest neighbours v of u of  sim(u, v) * x[v, j]
    where x is the binary positive user-item matrix.

    Dividing by sum |sim| (the lecture's normalisation) would not change any user's ranking, because
    the divisor is the same for every candidate item, so it is omitted. With binary feedback there
    is no rating scale to mean-centre (slide 11: normalisation becomes "activity bias").

    sim   'cosine'  : cosine of the (optionally item-weighted) binary rows
          'jaccard' : |intersection| / |union| of the two item sets
    idf   True      : item j is weighted by w_j = ln((U + 1) / (pop_j + 1)) inside the cosine, so that
                      blockbusters count for little (slide 11: "discount by item frequency").
                      Only meaningful with cosine.
    shrink gamma > 0: significance weighting, sim *= min(n_common, gamma) / gamma (slide 10), where
                      n_common is the number of co-liked items. 0 switches it off.
    k     number of neighbours kept per user (negative similarities do not occur with binary data).
    """

    def __init__(self, n_items, pos_threshold=4, k=50, sim="cosine", idf=False, shrink=0):
        if idf and sim != "cosine":
            raise ValueError("idf weighting is only defined here for cosine")
        self.n_items, self.thr, self.k, self.sim, self.idf, self.shrink = n_items, pos_threshold, k, sim, idf, shrink

    def fit(self, df, n_users):
        import scipy.sparse as sp

        X = interaction_matrix(df, n_users, self.n_items, True, self.thr)  # csr, float32, binary
        co = (X @ X.T).toarray().astype(np.float32)                        # co-liked item counts
        n_u = np.asarray(X.sum(1)).ravel().astype(np.float32)
        if self.sim == "cosine":
            if self.idf:
                pop = np.asarray(X.sum(0)).ravel()
                w = np.log((n_users + 1.0) / (pop + 1.0)).astype(np.float32)
                A = X.multiply(np.sqrt(w)[None, :]).tocsr()
                dot = (A @ A.T).toarray().astype(np.float32)
            else:
                dot = co
            norm = np.sqrt(np.maximum(np.diag(dot), 0))
            S = dot / np.maximum(np.outer(norm, norm), 1e-12)
        elif self.sim == "jaccard":
            S = co / np.maximum(n_u[:, None] + n_u[None, :] - co, 1e-12)
        else:
            raise ValueError(self.sim)
        if self.shrink > 0:
            S = S * (np.minimum(co, self.shrink) / self.shrink)
        np.fill_diagonal(S, 0.0)
        kk = int(min(self.k, S.shape[0] - 1))
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        vals = np.take_along_axis(S, idx, axis=1)
        rows = np.repeat(np.arange(n_users), kk)
        self.Sk = sp.csr_matrix((vals.ravel(), (rows, idx.ravel())), shape=(n_users, n_users))
        self.X = X
        pop_all = np.asarray(X.sum(0)).ravel()
        self.tb = pop_all / (pop_all.max() + 1.0)
        return self

    def scores(self, users):
        return (self.Sk[users] @ self.X).toarray().astype(np.float64) + 1e-9 * self.tb


class ImplicitALS:
    """Weighted ALS matrix factorisation for implicit feedback via the `implicit` library.

    Input: the binary positive matrix (rating >= threshold). `alpha` is the confidence given to
    observed positives (implicit multiplies the matrix values by alpha and gives every unobserved
    cell confidence 1, so alpha = 1 means no extra trust in the observed cells). `reg` is the L2
    regularisation, `factors` the number of latent factors k. Note: this is WRMF (Hu et al.),
    i.e. squared loss over all cells with confidence weights and no bias terms.
    """

    def __init__(self, n_items, pos_threshold=4, factors=64, reg=0.1, alpha=10.0, iterations=15, seed=42):
        self.n_items, self.thr = n_items, pos_threshold
        self.factors, self.reg, self.alpha, self.iterations, self.seed = factors, reg, alpha, iterations, seed

    def fit(self, df, n_users):
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError as e:  # pragma: no cover
            raise ImportError("ImplicitALS needs the 'implicit' package (pip install implicit)") from e
        X = interaction_matrix(df, n_users, self.n_items, True, self.thr)
        model = AlternatingLeastSquares(factors=self.factors, regularization=self.reg, alpha=self.alpha,
                                        iterations=self.iterations, random_state=self.seed,
                                        calculate_training_loss=False)
        model.fit(X, show_progress=False)
        self.U = np.asarray(model.user_factors, dtype=np.float64)
        self.V = np.asarray(model.item_factors, dtype=np.float64)
        return self

    def scores(self, users):
        return self.U[users] @ self.V.T
