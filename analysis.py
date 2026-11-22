"""Helpers for the Part 2 analyses: paired comparisons, disaggregation groups, subsampling."""
import numpy as np

import metrics


def paired_diff_ci(a, b, n_boot=1000, seed=42):
    """Mean of the per-user difference a - b with a 95% percentile bootstrap CI (users resampled).
    a and b are per-user metric vectors over the SAME users, so this is a paired comparison."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    lo, hi = metrics.bootstrap_ci(d, n_boot, seed)
    return float(d.mean()), lo, hi


def activity_groups(fit_df, users, n_users):
    """Low vs high activity users: split at the median number of ratings in the fitting data.
    Returns boolean masks over `users` and the median used."""
    counts = np.bincount(fit_df["u"].to_numpy(), minlength=n_users)[users]
    med = float(np.median(counts))
    return counts <= med, counts > med, med


def item_groups(fit_df, n_items, head_frac=0.2):
    """Popular ('head') vs long-tail items: the head_frac most-rated items in the fitting data.
    Returns the boolean head mask over items and the smallest rating count inside the head."""
    cnt = np.bincount(fit_df["i"].to_numpy(), minlength=n_items)
    order = np.argsort(-cnt, kind="stable")
    n_head = int(round(head_frac * n_items))
    head = np.zeros(n_items, dtype=bool)
    head[order[:n_head]] = True
    return head, int(cnt[order[n_head - 1]])


def item_group_metrics(topk, rel_dense, mask):
    """For one item group (mask over items):
    - per-user recall@10 restricted to the user's relevant items inside the group
      (hits among top-10 that are in the group / number of relevant items in the group),
      defined only for users with at least one relevant item in the group (returned as `ok`);
    - the share of all recommended slots that go to items of the group."""
    hits = np.take_along_axis(rel_dense, topk, axis=1) & mask[topk]
    n_rel = (rel_dense & mask[None, :]).sum(1)
    ok = n_rel > 0
    rec = np.zeros(len(topk))
    rec[ok] = hits.sum(1)[ok] / n_rel[ok]
    return rec, ok, float(mask[topk].mean())


def subsample(df, frac, seed):
    """Keep a uniformly random fraction of the interactions (rows)."""
    return df if frac >= 1.0 else df.sample(frac=frac, random_state=seed)


def topk_metrics(topk, rel_dense, k=10):
    """Per-user precision@k, recall@k, NDCG@k from top-k lists and a dense boolean relevance matrix
    (same definitions as metrics.evaluate). `ok` marks users with at least one relevant item;
    the other users get zeros and must be dropped by the caller."""
    hits = np.take_along_axis(rel_dense, topk, axis=1)
    n_rel = rel_dense.sum(1)
    ok = n_rel > 0
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    idcg = np.cumsum(disc)
    safe = np.maximum(n_rel, 1)
    prec = hits.sum(1) / k
    rec = np.where(ok, hits.sum(1) / safe, 0.0)
    ndcg = np.where(ok, (hits * disc).sum(1) / idcg[np.minimum(safe, k) - 1], 0.0)
    return prec, rec, ndcg, ok
