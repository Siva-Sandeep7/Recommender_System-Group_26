"""Reproduces every number and figure for Part 1.

    python run_all.py --data-dir /path/to/ml-1m
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data
import eda
import metrics
import models
import report
import results

K = 10
SEED = 42


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder containing ratings.dat, movies.dat, users.dat")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--pos-threshold", type=int, default=4, help="rating >= this counts as positive/relevant")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-mode", choices=["global", "per_user"], default="per_user",
                    help="global: one cut date for everyone; per_user: chronological split inside each user's history")
    ap.add_argument("--min-user", type=int, default=0, help="main-run filter: min ratings per user")
    ap.add_argument("--min-item", type=int, default=0, help="main-run filter: min ratings per item")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--quick", action="store_true", help="tiny grids, for smoke tests only")
    return ap.parse_args()


def make_model(spec, n_items, thr):
    kind, p = spec
    return models.MostPopular(n_items, thr, **p) if kind == "pop" else models.ItemKNN(n_items, thr, **p)


def prepare_eval(fit_df, target_df, split, thr):
    """Users to evaluate: warm (history in fit data) with >=1 positive target item."""
    n_u, n_i = split.n_users, split.n_items
    warm = np.zeros(n_u, bool)
    warm[fit_df["u"].unique()] = True
    has_rel = np.zeros(n_u, bool)
    has_rel[target_df.loc[target_df["r"] >= thr, "u"].unique()] = True
    users = np.where(warm & has_rel)[0]
    tpos = target_df[target_df["r"] >= thr]
    known_items = np.zeros(n_i, bool)
    known_items[fit_df["i"].unique()] = True
    info = dict(
        n_users_with_positive_target=int(has_rel.sum()),
        n_cold_users_excluded=int((has_rel & ~warm).sum()),
        n_eval_users=int(len(users)),
        n_positive_target_interactions=int(len(tpos)),
        pct_positive_target_on_items_unseen_in_fit=float(100 * np.mean(~known_items[tpos["i"].to_numpy()])),
    )
    return dict(
        users=users,
        seen=data.interaction_matrix(fit_df, n_u, n_i, False, thr),
        rel=data.interaction_matrix(target_df, n_u, n_i, True, thr),
        info=info,
    )


def run(spec, fit_df, ev, split, thr):
    m = make_model(spec, split.n_items, thr).fit(fit_df, split.n_users)
    return metrics.evaluate(m, ev["users"], ev["seen"], ev["rel"], split.n_items, K)


def tune(split, ev_val, thr, quick):
    """Grid search on the validation set only. Returns the log and the best spec per family."""
    log, best = [], {}
    wins = [30, 90] if quick else [7, 30, 90, 180, 365]
    families = {
        "MostPop (all-time, all ratings)": dict(positive_only=False, windows=[None]),
        "MostPop (all-time, positive only)": dict(positive_only=True, windows=[None]),
        "MostPop (recent window, all ratings)": dict(positive_only=False, windows=wins),
        "MostPop (recent window, positive only)": dict(positive_only=True, windows=wins),
    }
    for fam, fp in families.items():
        for w in fp["windows"]:
            spec = ("pop", dict(positive_only=fp["positive_only"], window_days=w))
            res = run(spec, split.train, ev_val, split, thr)
            log.append(dict(family=fam, config=json.dumps(spec[1]), **res["summary"]))
            if fam not in best or res["summary"][f"NDCG@{K}"] > best[fam][1]:
                best[fam] = (spec, res["summary"][f"NDCG@{K}"])
    sims = ["cosine"] if quick else ["cosine", "jaccard", "adjcos"]
    ks = [20, 50] if quick else [10, 20, 50, 100, 200, 500]
    fam = "ItemKNN"
    for sim in sims:
        m = models.ItemKNN(split.n_items, thr, k=ks[0], sim=sim).fit(split.train, split.n_users)
        for k in ks:
            m.set_k(k)
            res = metrics.evaluate(m, ev_val["users"], ev_val["seen"], ev_val["rel"], split.n_items, K)
            spec = ("knn", dict(sim=sim, k=k))
            log.append(dict(family=fam, config=json.dumps(spec[1]), **res["summary"]))
            if fam not in best or res["summary"][f"NDCG@{K}"] > best[fam][1]:
                best[fam] = (spec, res["summary"][f"NDCG@{K}"])
    return log, {f: v[0] for f, v in best.items()}


def split_stats(split, ev_val, ev_test):
    def d(ts):
        return "n/a (per-user split)" if ts is None else str(pd.to_datetime(ts, unit="s"))
    n = len(split.train) + len(split.val) + len(split.test)
    return dict(
        n_users=split.n_users, n_items=split.n_items,
        t_train_end=d(split.t_train_end), t_val_end=d(split.t_val_end),
        n_train=len(split.train), n_val=len(split.val), n_test=len(split.test),
        frac_train=len(split.train) / n, frac_val=len(split.val) / n, frac_test=len(split.test) / n,
        train_start=d(split.train["ts"].min()), test_end=d(split.test["ts"].max()),
        val_eval=ev_val["info"], test_eval=ev_test["info"],
    )


def main():
    args = parse()
    thr = args.pos_threshold
    np.random.seed(args.seed)
    out = Path(args.out_dir)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("[1/6] loading data")
    raw, movies, users = data.load_movielens(args.data_dir)
    filt, filt_info = data.filter_min_counts(raw, args.min_user, args.min_item)
    split = data.temporal_split(filt, args.train_frac, args.val_frac, args.split_mode)

    print("[2/6] data characterization")
    cut = None if split.t_train_end is None else (split.t_train_end, split.t_val_end)
    stats = eda.describe(raw, movies, users, fig_dir, cutoffs=cut)

    trainval = pd.concat([split.train, split.val])
    ev_val = prepare_eval(split.train, split.val, split, thr)
    ev_test = prepare_eval(trainval, split.test, split, thr)
    sstats = split_stats(split, ev_val, ev_test)

    print("[3/6] tuning on validation (%d val users)" % len(ev_val["users"]))
    log, best = tune(split, ev_val, thr, args.quick)
    pd.DataFrame(log).to_csv(out / "tuning_log.csv", index=False)

    print("[4/6] test evaluation (refit on train+val)")
    rows, test_summary = [], {}
    for fam, spec in best.items():
        res = run(spec, trainval, ev_test, split, thr)
        test_summary[fam] = dict(config=spec[1], **res["summary"])
        cfg = json.dumps(spec[1])
        for name, key in [(f"Precision@{K}", "prec"), (f"Recall@{K}", "rec"), (f"NDCG@{K}", "ndcg")]:
            ci = metrics.bootstrap_ci(res[key], args.n_boot, args.seed)
            test_summary[fam][name + "_ci"] = ci
            rows.append(results.make_row(fam, cfg, "test", name, res["summary"][name], ci,
                                         len(ev_test["users"]), args.seed))
        rows.append(results.make_row(fam, cfg, "test", f"Coverage@{K}", res["summary"][f"Coverage@{K}"],
                                     (None, None), len(ev_test["users"]), args.seed))
    results.write_rows(rows, out / "results_long.csv")
    results.write_template(test_summary, vars(args), len(ev_test["users"]), split.n_items, out / "results.csv")

    print("[5/6] filtering sensitivity (best configs held fixed)")
    sens = []
    for t_item in ([0, 10] if args.quick else [0, 5, 10, 20]):
        f2, i2 = data.filter_min_counts(raw, args.min_user, t_item)
        s2 = data.temporal_split(f2, args.train_frac, args.val_frac, args.split_mode)
        tv2 = pd.concat([s2.train, s2.val])
        ev2 = prepare_eval(tv2, s2.test, s2, thr)
        for fam, spec in best.items():
            res = run(spec, tv2, ev2, s2, thr)
            sens.append(dict(min_item=t_item, rows_removed=i2["rows_removed"], pct_rows_removed=i2["pct_rows_removed"],
                             items_removed=i2["items_removed"], n_items=s2.n_items,
                             n_eval_users=len(ev2["users"]), family=fam, **res["summary"]))
    pd.DataFrame(sens).to_csv(out / "filter_sensitivity.csv", index=False)

    print("[6/6] writing summaries")
    summary = dict(args=vars(args), filter_info=filt_info, data_stats=stats, split=sstats,
                   best_configs={f: s[1] for f, s in best.items()}, test=test_summary,
                   n_tuning_configs=dict(
                       pop=sum(1 for r in log if r["family"].startswith("MostPop")),
                       knn=sum(1 for r in log if r["family"] == "ItemKNN")))
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    report.write_report_numbers(summary, pd.DataFrame(log), pd.DataFrame(sens), out / "report_numbers.md")
    print("done in %.0fs -> %s" % (time.time() - t0, out))


if __name__ == "__main__":
    main()
