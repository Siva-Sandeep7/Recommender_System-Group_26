"""Loading, filtering and temporal splitting of MovieLens 1M."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

SEC_PER_DAY = 86400


def load_movielens(data_dir):
    """Parse the '::'-separated MovieLens 1M files."""
    d = Path(data_dir)
    kw = dict(sep="::", engine="python", encoding="latin-1", header=None)
    ratings = pd.read_csv(d / "ratings.dat", names=["user", "item", "rating", "ts"], **kw)
    movies = pd.read_csv(d / "movies.dat", names=["item", "title", "genres"], **kw)
    users = pd.read_csv(d / "users.dat", names=["user", "gender", "age", "occupation", "zip"], **kw)
    return ratings, movies, users


def filter_min_counts(ratings, min_user=0, min_item=0):
    """Iteratively drop users/items with fewer than the given number of ratings.

    Returns the filtered frame and a dict describing exactly what was removed.
    """
    df = ratings
    while True:
        n_before = len(df)
        if min_item > 0:
            c = df["item"].value_counts()
            df = df[df["item"].isin(c.index[c >= min_item])]
        if min_user > 0:
            c = df["user"].value_counts()
            df = df[df["user"].isin(c.index[c >= min_user])]
        if len(df) == n_before:
            break
    info = dict(
        min_user=int(min_user),
        min_item=int(min_item),
        rows_before=int(len(ratings)),
        rows_after=int(len(df)),
        rows_removed=int(len(ratings) - len(df)),
        pct_rows_removed=float(100 * (len(ratings) - len(df)) / len(ratings)),
        users_removed=int(ratings["user"].nunique() - df["user"].nunique()),
        items_removed=int(ratings["item"].nunique() - df["item"].nunique()),
    )
    return df.copy(), info


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    t_train_end: int
    t_val_end: int
    user_ids: np.ndarray
    item_ids: np.ndarray


def temporal_split(ratings, train_frac=0.8, val_frac=0.1, mode="global"):
    """Temporal split.

    mode="global":   cut the timeline at the train_frac and train_frac+val_frac quantiles
                     of the rating timestamps. Up to the first cut is train, between the
                     cuts is validation, after the second cut is test.
    mode="per_user": inside every user's own chronological history, the earliest
                     train_frac of ratings are train, the next val_frac validation, the
                     rest test. Keeps every user warm (no cold-start users), at the cost
                     of no single global cut date.

    Users/items are re-indexed to 0..n-1 over the whole (filtered) dataset, so the
    candidate catalogue is the same for all splits.
    """
    df = ratings.copy()
    uids = np.sort(df["user"].unique())
    iids = np.sort(df["item"].unique())
    df["u"] = np.searchsorted(uids, df["user"].to_numpy())
    df["i"] = np.searchsorted(iids, df["item"].to_numpy())
    df = df.rename(columns={"rating": "r"})[["u", "i", "r", "ts"]].reset_index(drop=True)
    if mode == "per_user":
        df = df.sort_values(["u", "ts", "i"], kind="stable").reset_index(drop=True)
        pos = df.groupby("u").cumcount().to_numpy()
        n = df.groupby("u")["u"].transform("size").to_numpy()
        n_tr = np.floor(n * train_frac + 1e-9)
        n_tr2 = np.floor(n * (train_frac + val_frac) + 1e-9)
        train = df[pos < n_tr]
        val = df[(pos >= n_tr) & (pos < n_tr2)]
        test = df[pos >= n_tr2]
        return Split(train, val, test, len(uids), len(iids), None, None, uids, iids)
    if mode != "global":
        raise ValueError(mode)
    ts = df["ts"].to_numpy()
    t1 = int(np.quantile(ts, train_frac, method="lower"))
    t2 = int(np.quantile(ts, train_frac + val_frac, method="lower"))
    train = df[df["ts"] <= t1]
    val = df[(df["ts"] > t1) & (df["ts"] <= t2)]
    test = df[df["ts"] > t2]
    return Split(train, val, test, len(uids), len(iids), t1, t2, uids, iids)


def interaction_matrix(df, n_users, n_items, positive_only=False, pos_threshold=4):
    """Binary user x item CSR matrix. positive_only keeps rating >= pos_threshold."""
    d = df[df["r"] >= pos_threshold] if positive_only else df
    m = sp.csr_matrix(
        (np.ones(len(d), dtype=np.float32), (d["u"].to_numpy(), d["i"].to_numpy())),
        shape=(n_users, n_items),
    )
    m.data[:] = 1.0
    return m
