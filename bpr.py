"""SELF-IMPLEMENTED ALGORITHM (Part 2): Bayesian Personalised Ranking, matrix-factorisation
variant (BPR-MF). Lecture 2.1, slide 12 (Rendle et al., 2009). Written from the lecture
formulas with NumPy only; no recommender library is used.

Model
    score(u, i) = p_u . q_i + b_i                      (p_u, q_i in R^k, b_i = item bias)
    A user-bias term would cancel in every difference below, so it is not needed.

Training data are triples (u, i, j): u has a positive interaction with i (rating >= threshold)
and has NO interaction at all with j (j is a sampled negative).

Objective (maximised, summed over sampled triples)
    ln sigma( x_uij )  -  (reg / 2) * ( |p_u|^2 + |q_i|^2 + |q_j|^2 + b_i^2 + b_j^2 )
    x_uij = score(u, i) - score(u, j),   sigma(x) = 1 / (1 + exp(-x))
The lecture writes the penalty as lambda * |Theta|^2; here reg = 2 * lambda.

Gradients of one triple, with g = sigma(-x_uij) = 1 - sigma(x_uij):
    d/d p_u = g * (q_i - q_j) - reg * p_u
    d/d q_i = g * p_u         - reg * q_i
    d/d q_j = -g * p_u        - reg * q_j
    d/d b_i = g               - reg * b_i
    d/d b_j = -g              - reg * b_j
Each parameter moves by  lr * gradient  (stochastic gradient ASCENT).

Optimisation: mini-batch SGD. All triples of a batch are computed from the same parameter
values and their updates are summed (np.add.at, so repeated users/items accumulate exactly).
batch_size = 1 is the textbook one-triple-at-a-time SGD.
"""
import numpy as np


# ---------------------------------------------------------------- the maths, kept separate
def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # numerically stable


def triple_objective(p_u, q_i, q_j, b_i, b_j, reg):
    """ln sigma(x_uij) - reg/2 * (squared norms of the parameters involved). Single triple."""
    x = p_u @ (q_i - q_j) + b_i - b_j
    pen = 0.5 * reg * (p_u @ p_u + q_i @ q_i + q_j @ q_j + b_i ** 2 + b_j ** 2)
    return -np.logaddexp(0.0, -x) - pen  # ln sigma(x) = -ln(1 + exp(-x))


def triple_grads(p_u, q_i, q_j, b_i, b_j, reg):
    """Analytic gradients of triple_objective. Works row-wise on batches:
    p_u, q_i, q_j have shape (B, k); b_i, b_j have shape (B,). Returns
    (g_p, g_qi, g_qj, g_bi, g_bj, x) where x are the margins x_uij."""
    x = np.einsum("bk,bk->b", p_u, q_i - q_j) + b_i - b_j
    g = sigmoid(-x)[:, None]                     # sigma(-x)
    g_p = g * (q_i - q_j) - reg * p_u
    g_qi = g * p_u - reg * q_i
    g_qj = -g * p_u - reg * q_j
    g_bi = g[:, 0] - reg * b_i
    g_bj = -g[:, 0] - reg * b_j
    return g_p, g_qi, g_qj, g_bi, g_bj, x


# ---------------------------------------------------------------------------- the model
class BPR:
    def __init__(self, n_items, pos_threshold=4, k=32, lr=0.05, reg=0.01, epochs=20,
                 batch_size=1024, neg_alpha=0.0, init_std=0.1, seed=42):
        """neg_alpha = 0: negatives j drawn uniformly from the items u has not touched
        (the plain BPR of the lecture). neg_alpha > 0: j is drawn with probability
        proportional to popularity**neg_alpha among all items, then re-drawn if u has
        touched it (a popularity-aware negative sampler)."""
        self.n_items, self.thr = n_items, pos_threshold
        self.k, self.lr, self.reg, self.epochs = k, lr, reg, epochs
        self.batch_size, self.neg_alpha, self.init_std, self.seed = batch_size, neg_alpha, init_std, seed

    # -- negative sampling ----------------------------------------------------------------
    def _draw(self, size, rng):
        if self.neg_alpha == 0:
            return rng.integers(0, self.n_items, size=size)
        return np.searchsorted(self._cdf, rng.random(size), side="right").clip(0, self.n_items - 1)

    def _sample_negatives(self, u, rng):
        """One negative per row of u: an item with no interaction (any rating) of that user in
        the fitting data. Rejection sampling; collisions are re-drawn until none remain."""
        j = self._draw(len(u), rng)
        bad = self._touched[u, j]
        while bad.any():
            j[bad] = self._draw(int(bad.sum()), rng)
            bad = self._touched[u, j]
        return j

    # -- training -------------------------------------------------------------------------
    def fit(self, df, n_users):
        rng = np.random.default_rng(self.seed)
        n_items, k = self.n_items, self.k
        # "touched" = rated at all (positive or not): never used as a negative, and consistent
        # with the evaluation protocol, which also removes every rated item from the ranking.
        self._touched = np.zeros((n_users, n_items), dtype=bool)
        self._touched[df["u"].to_numpy(), df["i"].to_numpy()] = True
        pos = df[df["r"] >= self.thr]
        pu, pi = pos["u"].to_numpy(), pos["i"].to_numpy()
        if self.neg_alpha > 0:
            pop = np.bincount(pi, minlength=n_items).astype(np.float64) + 1.0
            w = pop ** self.neg_alpha
            self._cdf = np.cumsum(w / w.sum())
        self.P = rng.normal(0.0, self.init_std, (n_users, k))
        self.Q = rng.normal(0.0, self.init_std, (n_items, k))
        self.b = np.zeros(n_items)
        self.history_ = []  # mean ln sigma(x) of the sampled triples in each epoch (should rise)
        n = len(pu)
        for _ in range(self.epochs):
            order = rng.permutation(n)
            total = 0.0
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                u, i = pu[idx], pi[idx]
                j = self._sample_negatives(u, rng)
                g_p, g_qi, g_qj, g_bi, g_bj, x = triple_grads(
                    self.P[u], self.Q[i], self.Q[j], self.b[i], self.b[j], self.reg)
                total += float(-np.logaddexp(0.0, -x).sum())
                lr = self.lr
                np.add.at(self.P, u, lr * g_p)
                np.add.at(self.Q, i, lr * g_qi)
                np.add.at(self.Q, j, lr * g_qj)
                np.add.at(self.b, i, lr * g_bi)
                np.add.at(self.b, j, lr * g_bj)
            self.history_.append(total / n)
        return self

    def scores(self, users):
        return self.P[users] @ self.Q.T + self.b[None, :]
