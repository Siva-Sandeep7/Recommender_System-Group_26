"""Part 2 pipeline: five models under one protocol, equal tuning budget, multiple seeds,
paired confidence intervals, disaggregated results and a sparsity ablation.

    python run_part2.py --data-dir /path/to/ml-1m

Models: Most Popular and Item-kNN (the Part 1 baselines, re-tuned here with the same budget),
User-kNN, implicit-feedback ALS (library `implicit`), and two self-implemented models:
BPR (bpr.py) and EASE (ease.py, closed-form linear item-item autoencoder).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import analysis
import bpr
import data
import ease
import metrics
import models
import models_p2
from run_all import prepare_eval

K = 10
SEED = 42
NAMES = {
    "pop": "Most Popular", "iknn": "Item-based kNN", "uknn": "User-based kNN",
    "als": "ALS (implicit, library)", "bpr": "BPR (self-implemented)", "ease": "EASE (self-implemented)",
}
PARADIGM = {
    "pop": "non-personalised baseline", "iknn": "neighbourhood CF (Part 1 baseline)",
    "uknn": "neighbourhood CF", "als": "latent-factor CF", "bpr": "ranking-loss CF",
    "ease": "linear item-item autoencoder (closed form)",
}
STOCHASTIC = {"pop": False, "iknn": False, "uknn": False, "als": True, "bpr": True, "ease": False}
T0 = time.time()


def log(msg):
    print("[%5.0fs] %s" % (time.time() - T0, msg), flush=True)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="outputs_part2")
    ap.add_argument("--split-mode", choices=["global", "per_user"], default="per_user")
    ap.add_argument("--pos-threshold", type=int, default=4)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--budget", type=int, default=18, help="validation configurations per model family")
    ap.add_argument("--n-seeds", type=int, default=5, help="seeds for stochastic models and subsampling")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--skip-als", action="store_true", help="skip ALS if the implicit package is unavailable")
    ap.add_argument("--reuse-tuning", action="store_true", help="load best_configs.json from out-dir instead of tuning")
    ap.add_argument("--quick", action="store_true", help="tiny budgets, for smoke tests only")
    return ap.parse_args()


# ------------------------------------------------------------------------------ search spaces
def grid_of(**axes):
    names = list(axes)
    out = [{}]
    for n in names:
        out = [dict(c, **{n: v}) for c in out for v in axes[n]]
    return out


def spaces(quick):
    pop = [dict(positive_only=p, window_days=None) for p in (False, True)] + [
        dict(positive_only=p, window_days=w) for p in (False, True) for w in ((30, 180) if quick else (7, 30, 90, 180, 365))]
    iknn = grid_of(sim=["cosine"] if quick else ["cosine", "jaccard", "adjcos"], k=[20, 50] if quick else [10, 20, 50, 100, 200, 500])
    uknn = [dict(sim=s, idf=i, k=k, shrink=g) for (s, i) in (("cosine", False), ("cosine", True), ("jaccard", False))
            for k in ([20, 50] if quick else [20, 50, 100, 200, 400]) for g in (0, 30)]
    als = grid_of(factors=[8, 16] if quick else [16, 32, 64, 128], reg=[0.1] if quick else [0.01, 0.1, 1.0],
                  alpha=[5, 20] if quick else [5, 20, 60], iterations=[5 if quick else 15])
    bp = grid_of(k=[8, 16] if quick else [16, 32, 64], lr=[0.05] if quick else [0.02, 0.05, 0.1],
                 reg=[0.01] if quick else [0.001, 0.01, 0.05], epochs=[2, 4] if quick else [15, 30])
    ea = [dict(lam=l) for l in ([100, 1000] if quick else [10, 25, 50, 100, 200, 350, 500, 750, 1000, 1500, 2000, 3000, 5000])]
    return dict(pop=pop, iknn=iknn, uknn=uknn, als=als, bpr=bp, ease=ea)


def sample_configs(grid, budget, seed):
    """All configurations if the grid fits in the budget, else a seeded random subset."""
    if len(grid) <= budget:
        return grid
    idx = np.sort(np.random.default_rng(seed).choice(len(grid), size=budget, replace=False))
    return [grid[i] for i in idx]


def build(fam, cfg, seed, n_items, thr):
    if fam == "pop":
        return models.MostPopular(n_items, thr, **cfg)
    if fam == "iknn":
        return models.ItemKNN(n_items, thr, **cfg)
    if fam == "uknn":
        return models_p2.UserKNN(n_items, thr, **cfg)
    if fam == "als":
        return models_p2.ImplicitALS(n_items, thr, seed=seed, **cfg)
    if fam == "bpr":
        return bpr.BPR(n_items, thr, seed=seed, **cfg)
    if fam == "ease":
        return ease.EASE(n_items, thr, **cfg)
    raise ValueError(fam)


def fit_eval(fam, cfg, seed, fit_df, ev, split, thr):
    m = build(fam, cfg, seed, split.n_items, thr).fit(fit_df, split.n_users)
    return metrics.evaluate(m, ev["users"], ev["seen"], ev["rel"], split.n_items, K)


# ------------------------------------------------------------------------------------- tuning
def tune(fams, split, ev_val, thr, args, seeds):
    log_rows, best = [], {}
    sp = spaces(args.quick)
    for fam in fams:
        grid = sample_configs(sp[fam], args.budget, SEED)
        log("tuning %-26s %d of %d grid points" % (NAMES[fam], len(grid), len(sp[fam])))
        for cfg in grid:
            t = time.time()
            res = fit_eval(fam, cfg, seeds[0], split.train, ev_val, split, thr)
            s = res["summary"]
            log_rows.append(dict(family=NAMES[fam], key=fam, config=json.dumps(cfg), **s, fit_eval_seconds=round(time.time() - t, 1)))
            if fam not in best or s["NDCG@10"] > best[fam][1]:
                best[fam] = (cfg, s["NDCG@10"])
        log("   best validation NDCG@10 %.4f  %s" % (best[fam][1], best[fam][0]))
    return log_rows, {f: v[0] for f, v in best.items()}


# ------------------------------------------------------------------------------ summarising
def md_table(df, floatfmt="%.4f"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join((floatfmt % v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def summarise(runs, n_boot):
    """Aggregate the runs (one per seed) of one model."""
    keys = ["Precision@10", "Recall@10", "NDCG@10", "Coverage@10"]
    vec = {k: np.mean([r[v] for r in runs], axis=0) for k, v in [("Precision@10", "prec"), ("Recall@10", "rec"), ("NDCG@10", "ndcg")]}
    row = {}
    for k in keys:
        per = np.array([r["summary"][k] for r in runs])
        row[k] = float(per.mean())
        row[k + " std over seeds"] = float(per.std(ddof=1)) if len(per) > 1 else 0.0
    for k in keys[:3]:
        lo, hi = metrics.bootstrap_ci(vec[k], n_boot, SEED)
        row[k + " CI low"], row[k + " CI high"] = lo, hi
    return row, vec


def main():
    args = parse()
    thr = args.pos_threshold
    seeds = [SEED + i for i in range(2 if args.quick else args.n_seeds)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fams = [f for f in ("pop", "iknn", "uknn", "als", "bpr", "ease") if not (f == "als" and args.skip_als)]

    log("loading data")
    raw, movies, users_df = data.load_movielens(args.data_dir)
    split = data.temporal_split(raw, args.train_frac, args.val_frac, args.split_mode)
    trainval = pd.concat([split.train, split.val])
    ev_val = prepare_eval(split.train, split.val, split, thr)
    ev_test = prepare_eval(trainval, split.test, split, thr)
    log("split=%s  validation users %d  test users %d  items %d" %
        (args.split_mode, len(ev_val["users"]), len(ev_test["users"]), split.n_items))

    # ---- 1. tuning on validation only, same budget for every family --------------------------
    best_path = out / "best_configs.json"
    if args.reuse_tuning and best_path.exists():
        best = json.load(open(best_path))
        log("reusing tuning results from %s" % best_path)
        tuning_rows = None
        missing = [f for f in fams if f not in best]
        if missing:  # families added after the saved tuning run are tuned now, the others keep their saved config
            new_rows, new_best = tune(missing, split, ev_val, thr, args, seeds)
            best.update(new_best)
            log_path = out / "tuning_log_part2.csv"
            old = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
            tuning_df = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True)
            tuning_df.to_csv(log_path, index=False)
            json.dump(best, open(best_path, "w"), indent=2)
            tuning_rows = tuning_df.to_dict("records")
        elif (out / "tuning_log_part2.csv").exists():
            tuning_rows = pd.read_csv(out / "tuning_log_part2.csv").to_dict("records")
    else:
        tuning_rows, best = tune(fams, split, ev_val, thr, args, seeds)
        pd.DataFrame(tuning_rows).to_csv(out / "tuning_log_part2.csv", index=False)
        json.dump(best, open(best_path, "w"), indent=2)

    # ---- 2. final test runs: refit on train+val, all seeds for stochastic models --------------
    log("final test evaluation")
    final, rows = {}, []
    for fam in fams:
        runs = []
        for s in (seeds if STOCHASTIC[fam] else seeds[:1]):
            runs.append(fit_eval(fam, best[fam], s, trainval, ev_test, split, thr))
        row, vec = summarise(runs, args.n_boot)
        final[fam] = dict(runs=runs, vec=vec)
        rows.append(dict(model=NAMES[fam], paradigm=PARADIGM[fam], config=json.dumps(best[fam]), n_seeds=len(runs),
                         n_test_users=len(ev_test["users"]), **row))
        log("   %-26s NDCG@10 %.4f [%.4f, %.4f]  coverage %.3f" %
            (NAMES[fam], row["NDCG@10"], row["NDCG@10 CI low"], row["NDCG@10 CI high"], row["Coverage@10"]))
    models_df = pd.DataFrame(rows)
    models_df.to_csv(out / "results_part2_models.csv", index=False)

    # ---- 3. paired comparisons on NDCG@10 (same users, per-user differences) ------------------
    pair_rows = []
    for a in fams:
        for b in fams:
            if a == b:
                continue
            d, lo, hi = analysis.paired_diff_ci(final[a]["vec"]["NDCG@10"], final[b]["vec"]["NDCG@10"], args.n_boot, SEED)
            pair_rows.append(dict(model_a=NAMES[a], model_b=NAMES[b], mean_diff_ndcg=d, ci_low=lo, ci_high=hi,
                                  ci_excludes_zero=bool(lo > 0 or hi < 0)))
    pairs_df = pd.DataFrame(pair_rows)
    pairs_df.to_csv(out / "pairwise_ndcg.csv", index=False)

    # ---- 4. disaggregated results ---------------------------------------------------------------
    log("disaggregation")
    users = ev_test["users"]
    low, high, med = analysis.activity_groups(trainval, users, split.n_users)
    head, head_min = analysis.item_groups(trainval, split.n_items, 0.2)
    rel_dense = ev_test["rel"][users].toarray() > 0
    dis = []
    for fam in fams:
        v = final[fam]["vec"]
        for gname, mask in (("low activity (<= median)", low), ("high activity (> median)", high)):
            for met in ("NDCG@10", "Recall@10"):
                lo, hi = metrics.bootstrap_ci(v[met][mask], args.n_boot, SEED)
                dis.append(dict(model=NAMES[fam], dimension="user activity", group=gname, metric=met,
                                value=float(v[met][mask].mean()), ci_low=lo, ci_high=hi, n=int(mask.sum()), slot_share=np.nan))
        per_gr = {g: [] for g in ("popular items (head)", "long-tail items")}
        shares = {g: [] for g in per_gr}
        for r in final[fam]["runs"]:
            for g, m in (("popular items (head)", head), ("long-tail items", ~head)):
                rec, ok, share = analysis.item_group_metrics(r["topk"], rel_dense, m)
                per_gr[g].append(rec)
                shares[g].append(share)
        for g, m in (("popular items (head)", head), ("long-tail items", ~head)):
            # which users count depends only on the test set and the group, not on the model
            _, ok, _ = analysis.item_group_metrics(final[fam]["runs"][0]["topk"], rel_dense, m)
            vec = np.mean(per_gr[g], axis=0)[ok]
            lo, hi = metrics.bootstrap_ci(vec, args.n_boot, SEED)
            dis.append(dict(model=NAMES[fam], dimension="item popularity", group=g, metric="Recall@10 (relevant items of the group)",
                            value=float(vec.mean()), ci_low=lo, ci_high=hi, n=int(ok.sum()), slot_share=float(np.mean(shares[g]))))
    dis_df = pd.DataFrame(dis)
    dis_df.to_csv(out / "results_part2_disaggregated.csv", index=False)

    # same breakdowns in the layout of the workbook tab "Part 2 - Disaggregated" (all four metrics per group)
    wb_rows = []
    allitems = np.ones(split.n_items, dtype=bool)
    for fam in fams:
        runs = final[fam]["runs"]
        seed_txt = "%d-%d (mean)" % (seeds[0], seeds[-1]) if STOCHASTIC[fam] else "n/a (deterministic)"
        groups = [
            ("User activity", "low activity", "<= %.0f ratings in train+validation (median split)" % med, low, allitems, "users"),
            ("User activity", "high activity", "> %.0f ratings in train+validation (median split)" % med, high, allitems, "users"),
            ("Item popularity", "popular items (head)", "%d most-rated items in train+validation (top 20%%, >= %d ratings)" % (head.sum(), head_min), None, head, "items"),
            ("Item popularity", "long-tail items", "all other %d items" % int((~head).sum()), None, ~head, "items"),
        ]
        for dim, gname, defn, umask, imask, kind in groups:
            per, covs, shares = [], [], []
            for r_ in runs:
                tk = r_["topk"]
                if kind == "users":
                    p, rc, nd, ok = analysis.topk_metrics(tk[umask], rel_dense[umask] & imask[None, :])
                    covs.append(len(np.unique(tk[umask])) / split.n_items)
                    shares.append(np.nan)
                else:
                    p, rc, nd, ok = analysis.topk_metrics(tk, rel_dense & imask[None, :])
                    covs.append(len(np.unique(tk[ok][imask[tk[ok]]])) / int(imask.sum()))
                    shares.append(float(imask[tk[ok]].mean()))
                per.append((p[ok], rc[ok], nd[ok]))
            P, R, N = (np.mean([x[i] for x in per], axis=0) for i in range(3))
            lo, hi = metrics.bootstrap_ci(N, args.n_boot, SEED)
            if kind == "users":
                note = "%d test users; NDCG@10 95%% CI [%.4f, %.4f]. Coverage = distinct items in these users' top-10 lists / %d catalogue items." % (len(N), lo, hi, split.n_items)
            else:
                note = ("%d test users with >= 1 relevant item in the group; only relevant items of the group count as hits (precision, recall and NDCG "
                        "computed against them); NDCG@10 95%% CI [%.4f, %.4f]; share of top-10 slots in the group %.3f; coverage = distinct group items "
                        "recommended / %d items in the group." % (len(N), lo, hi, np.mean(shares), int(imask.sum())))
            wb_rows.append({"Model": NAMES[fam], "Breakdown dimension": dim, "Group": gname, "Definition / threshold": defn,
                            "Precision@10": round(float(P.mean()), 6), "Recall@10": round(float(R.mean()), 6),
                            "NDCG@10": round(float(N.mean()), 6), "Catalogue coverage": round(float(np.mean(covs)), 6),
                            "Seed(s)": seed_txt, "Notes": note})
    pd.DataFrame(wb_rows).to_csv(out / "workbook_part2_disaggregated.csv", index=False)

    # ---- 5. sparsity ablation ---------------------------------------------------------------------
    log("sparsity ablation")
    sp_rows = []
    for frac in (1.0, 0.5, 0.2):
        for fam in fams:
            if frac == 1.0:
                runs = final[fam]["runs"]
            else:
                runs = []
                for s in seeds:  # different subsample (and model seed) per seed
                    sub = analysis.subsample(trainval, frac, s)
                    runs.append(fit_eval(fam, best[fam], s, sub, ev_test, split, thr))
            keys = ["Precision@10", "Recall@10", "NDCG@10", "Coverage@10"]
            per = {k: np.array([r["summary"][k] for r in runs]) for k in keys}
            sp_rows.append(dict(model=NAMES[fam], fraction=frac, n_runs=len(runs),
                                **{k: float(per[k].mean()) for k in keys},
                                ndcg_std_over_runs=float(per["NDCG@10"].std(ddof=1)) if len(runs) > 1 else 0.0))
            log("   %3.0f%%  %-26s NDCG@10 %.4f" % (100 * frac, NAMES[fam], per["NDCG@10"].mean()))
    sp_df = pd.DataFrame(sp_rows)
    sp_df["rank_by_ndcg"] = sp_df.groupby("fraction")["NDCG@10"].rank(ascending=False, method="min").astype(int)
    sp_df.to_csv(out / "results_part2_sparsity.csv", index=False)

    # ---- 6. summaries -----------------------------------------------------------------------------------
    info = dict(args=vars(args), seeds=seeds, split_mode=args.split_mode, n_test_users=int(len(users)),
                activity_median_ratings=med, head_items=int(head.sum()), head_min_ratings=head_min,
                budget=args.budget, best_configs=best)
    json.dump(info, open(out / "summary_part2.json", "w"), indent=2, default=str)
    write_report_numbers(out / "report_numbers_part2.md", info, tuning_rows, models_df, pairs_df, dis_df, sp_df)
    log("done -> %s" % out)


def write_report_numbers(path, info, tuning_rows, models_df, pairs_df, dis_df, sp_df):
    L = ["# Part 2 - numbers for the report\n"]
    a = L.append
    a("## 1. Protocol recap\n")
    a(f"- Split: {info['split_mode']}; test users evaluated: {info['n_test_users']:,}; seeds: {info['seeds']}.")
    a(f"- Tuning budget: {info['budget']} validation configurations per model family (a family whose whole space is smaller is searched exhaustively); "
      f"selection metric NDCG@10 on validation only; stochastic models tuned with seed {info['seeds'][0]}.")
    a(f"- Disaggregation thresholds: low/high activity split at the median of {info['activity_median_ratings']:.0f} ratings in the fitting data (train+validation); "
      f"popular items = the {info['head_items']} most-rated items (top 20%, at least {info['head_min_ratings']} ratings), all others long tail.\n")
    if tuning_rows:
        t = pd.DataFrame(tuning_rows)
        a("## 2. Tuning (validation)\n")
        a(md_table(t.groupby("family").agg(configs=("config", "count"), best_val_NDCG=("NDCG@10", "max"),
                                           worst_val_NDCG=("NDCG@10", "min")).reset_index()))
        a("\nBest configurations: " + json.dumps(info["best_configs"]) + "\n")
    a("## 3. Overall test results (mean over seeds; 95% bootstrap CI over users; std over seeds)\n")
    show = models_df[["model", "n_seeds", "Precision@10", "Recall@10", "NDCG@10", "NDCG@10 CI low", "NDCG@10 CI high",
                      "NDCG@10 std over seeds", "Coverage@10"]]
    a(md_table(show) + "\n")
    a("## 4. Paired NDCG@10 differences (model_a minus model_b; CI over users)\n")
    a(md_table(pairs_df) + "\n")
    a("## 5. Disaggregated results\n")
    a(md_table(dis_df) + "\n")
    for dim, grp in dis_df.groupby("dimension"):
        for g, sub in grp.groupby("group"):
            top = sub.sort_values("value", ascending=False).iloc[0]
            a(f"- Auto-check: best model for {dim} / {g} ({top['metric']}): **{top['model']}** ({top['value']:.4f}).")
    overall_best = models_df.sort_values("NDCG@10", ascending=False).iloc[0]["model"]
    a(f"- Overall best on NDCG@10: **{overall_best}**. Any group above whose best model differs is a finding that qualifies the headline.\n")
    a("## 6. Sparsity ablation (mean NDCG@10 over runs; rank 1 = best)\n")
    a(md_table(sp_df) + "\n")
    order = {f: list(g.sort_values("NDCG@10", ascending=False)["model"]) for f, g in sp_df.groupby("fraction")}
    for f in sorted(order, reverse=True):
        a(f"- {int(100 * f)}% of training interactions, best to worst: " + " > ".join(order[f]))
    changed = order[1.0] != order[0.2]
    a(f"- Ranking changes between 100% and 20%: **{'yes' if changed else 'no'}**.\n")
    a("## 7. Still to write by hand\n")
    a("- Interpretation: why the ranking looks the way it does overall, per group and under sparsity; which differences have a CI that excludes zero.")
    a("- Contribution, AI-use disclosure, Part 3 track announcement, repository URL.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
