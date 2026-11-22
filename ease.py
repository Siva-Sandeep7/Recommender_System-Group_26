"""EASE - Embarrassingly Shallow Autoencoder (Steck, WWW 2019), written from scratch with NumPy.

IDEA IN ONE SENTENCE
    Predict every item from all the OTHER items a user liked, with one linear regression per item,
    and solve all those regressions at once with a closed-form formula.

THE MODEL
    X  is the binary user-item matrix (1 = the user rated the item >= threshold), shape U x I.
    B  is an item-item weight matrix, shape I x I. Column j of B holds the weights used to predict
       item j from the user's other items:   score(u, j) = sum_i X[u, i] * B[i, j]   ->  S = X @ B.

THE TRAINING PROBLEM  (ridge regression with one constraint)
        min_B   ||X - X B||_F^2  +  lambda * ||B||_F^2        subject to   diag(B) = 0
    * ||X - X B||^2 : reconstruct each user's row from the row itself ...
    * diag(B) = 0   : ... but an item may NOT be used to predict itself (else B = identity, which is
                      perfect on the training data and useless for recommending new items).
    * lambda        : ridge penalty, the only hyper-parameter. Large lambda = smoother, more popularity-like.

THE CLOSED-FORM SOLUTION  (Lagrange multipliers; derived in the paper)
        G = X^T X + lambda I              (item-item co-occurrence counts, plus the ridge term)
        P = G^-1
        B[i, j] = - P[i, j] / P[j, j]     for i != j,       B[j, j] = 0
    One matrix inverse of size I x I (about 3,700 x 3,700 on MovieLens 1M, a few seconds).

WHY IT IS A GOOD, EXPLAINABLE CHOICE
    * No random numbers: no seed, no learning rate, no epochs, no sampling. Same data -> same result.
    * Compared with item-kNN (also item-item): kNN sets the weight of i for j to a fixed similarity
      (cosine) and keeps k neighbours. EASE LEARNS the weights jointly, so correlated items share
      credit instead of being counted twice, and negative weights are allowed ("people who liked A
      but not B ...").
    * Compared with ALS/BPR (latent factors): EASE has no low-rank bottleneck, it keeps the full I x I matrix.

Interface (same as the other models): fit(df, n_users) and scores(users).
Reference: Harald Steck, "Embarrassingly Shallow Autoencoders for Sparse Data", WWW 2019.
"""
import numpy as np

from data import interaction_matrix
from models import _tie_break


class EASE:
    def __init__(self, n_items, pos_threshold=4, lam=500.0):
        self.n_items, self.thr, self.lam = n_items, pos_threshold, float(lam)

    def fit(self, df, n_users):
        X = interaction_matrix(df, n_users, self.n_items, True, self.thr)   # sparse binary, U x I
        self.X = X
        self.tb = _tie_break(df, self.n_items)
        G = (X.T @ X).toarray().astype(np.float64)                          # co-occurrence counts
        G[np.diag_indices_from(G)] += self.lam                              # + lambda * I
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))                                               # column j divided by -P[j, j]
        B[np.diag_indices_from(B)] = 0.0                                    # an item never predicts itself
        self.B = B.astype(np.float32)
        return self

    def scores(self, users):
        s = np.asarray(self.X[users] @ self.B, dtype=np.float64)
        return s + 1e-9 * self.tb      # only decides ties (e.g. users with no positive item)
