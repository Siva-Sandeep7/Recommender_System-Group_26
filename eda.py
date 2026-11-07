"""Data characterization for Part 1: statistics and figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n)


def _dist(x):
    x = np.asarray(x)
    return dict(min=int(x.min()), median=float(np.median(x)), mean=float(x.mean()), max=int(x.max()))


def describe(ratings, movies, users, fig_dir, cutoffs=None):
    """Return a dict of statistics and write figures into fig_dir.

    cutoffs: optional (t_train_end, t_val_end) unix timestamps drawn on the activity plot.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_r = len(ratings)
    n_u = ratings["user"].nunique()
    n_i = ratings["item"].nunique()
    per_user = ratings.groupby("user").size().to_numpy()
    per_item = ratings.groupby("item").size().to_numpy()
    item_sorted = np.sort(per_item)[::-1]
    cum = np.cumsum(item_sorted) / item_sorted.sum()

    dt = pd.to_datetime(ratings["ts"], unit="s")
    day = dt.dt.floor("D")
    per_day = day.value_counts().sort_values(ascending=False)
    span = ratings.groupby("user")["ts"].agg(lambda s: (s.max() - s.min()) / 86400.0).to_numpy()
    year_counts = dt.dt.year.value_counts().sort_index()
    month_counts = dt.dt.to_period("M").value_counts().sort_index()

    stats = dict(
        n_users=int(n_u),
        n_items_rated=int(n_i),
        n_items_in_movies_file=int(len(movies)),
        n_ratings=int(n_r),
        density=float(n_r / (n_u * n_i)),
        ratings_per_user=_dist(per_user),
        ratings_per_item=_dist(per_item),
        items_with_lt10_ratings_pct=float(100 * np.mean(per_item < 10)),
        items_with_lt5_ratings_pct=float(100 * np.mean(per_item < 5)),
        top1pct_items_share_of_ratings=float(cum[max(int(0.01 * n_i) - 1, 0)]),
        top10pct_items_share_of_ratings=float(cum[max(int(0.10 * n_i) - 1, 0)]),
        top20pct_items_share_of_ratings=float(cum[max(int(0.20 * n_i) - 1, 0)]),
        item_popularity_gini=gini(per_item),
        user_activity_gini=gini(per_user),
        rating_counts={int(k): int(v) for k, v in ratings["rating"].value_counts().sort_index().items()},
        rating_mean=float(ratings["rating"].mean()),
        pct_ratings_ge4=float(100 * np.mean(ratings["rating"] >= 4)),
        time_start=str(dt.min()),
        time_end=str(dt.max()),
        time_span_days=float((ratings["ts"].max() - ratings["ts"].min()) / 86400),
        ratings_per_year={int(k): int(v) for k, v in year_counts.items()},
        busiest_day=str(per_day.index[0].date()),
        busiest_day_share_pct=float(100 * per_day.iloc[0] / n_r),
        busiest_30_days_share_pct=float(100 * per_day.iloc[:30].sum() / n_r),
        users_all_ratings_within_1day_pct=float(100 * np.mean(span <= 1)),
        users_all_ratings_within_7days_pct=float(100 * np.mean(span <= 7)),
        user_rating_span_days_median=float(np.median(span)),
        gender_counts=users["gender"].value_counts().to_dict(),
        age_counts={int(k): int(v) for k, v in users["age"].value_counts().sort_index().items()},
    )

    # --- Figure 1: long tail ---
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].loglog(np.arange(1, n_i + 1), item_sorted)
    ax[0].set(xlabel="item rank", ylabel="# ratings", title="Item popularity (rank-frequency)")
    ax[1].hist(per_user, bins=np.logspace(np.log10(per_user.min()), np.log10(per_user.max()), 40))
    ax[1].set_xscale("log")
    ax[1].set(xlabel="ratings per user", ylabel="# users", title="User activity")
    ax[2].plot(np.arange(1, n_i + 1) / n_i, cum)
    ax[2].set(xlabel="fraction of items (most popular first)", ylabel="cumulative share of ratings",
              title="Concentration of ratings")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_long_tail.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: rating distribution ---
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    vc = ratings["rating"].value_counts().sort_index()
    ax.bar(vc.index, vc.values)
    ax.set(xlabel="rating", ylabel="# ratings", title="Rating distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_rating_dist.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: activity over time, with split cutoffs ---
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = month_counts.index.to_timestamp()
    ax.semilogy(x, month_counts.values, marker="o", ms=3)
    if cutoffs is not None:
        for t, lab in zip(cutoffs, ["train|val", "val|test"]):
            ax.axvline(pd.to_datetime(t, unit="s"), color="r", ls="--", lw=1)
            ax.text(pd.to_datetime(t, unit="s"), ax.get_ylim()[1], lab, rotation=90, va="top", ha="right", fontsize=8)
    ax.set(xlabel="month rating was entered", ylabel="# ratings (log)", title="Activity over time")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_activity_time.png", dpi=150)
    plt.close(fig)
    return stats
