"""Results writers.

results.csv follows the column layout of the course workbook tab "Part 1 - Baselines"
(Model, Variant / settings, Similarity, k, Precision@10, Recall@10, NDCG@10,
Catalogue coverage, Seed, Notes). Column names must not be renamed or dropped.
A long-format record with confidence intervals is also written to results_long.csv.
"""
import csv

TEMPLATE_COLUMNS = ["Model", "Variant / settings", "Similarity", "k", "Precision@10", "Recall@10",
                    "NDCG@10", "Catalogue coverage", "Seed", "Notes"]

LONG_COLUMNS = ["model", "config", "split", "metric", "value", "ci_low", "ci_high", "n_eval_users", "seed"]


def make_row(model, config, split, metric, value, ci=(None, None), n_users=None, seed=None):
    return dict(model=model, config=config, split=split, metric=metric, value=value,
                ci_low=ci[0], ci_high=ci[1], n_eval_users=n_users, seed=seed)


def write_rows(rows, path, columns=LONG_COLUMNS):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _variant_text(cfg):
    if "sim" in cfg:
        return (f"{cfg['sim']} similarity on binary positive feedback (rating >= threshold), "
                f"top-{cfg['k']} neighbours per target item, scores summed over the user's positive history")
    scope = "all-time" if cfg["window_days"] is None else f"last {cfg['window_days']} days before the end of the fitted data"
    what = "positive ratings only" if cfg["positive_only"] else "all ratings"
    return f"{scope}; counts {what}"


def template_rows(test_summary, args, n_users, n_items):
    """One row per final test result, in the course template layout."""
    rows = []
    for fam, r in test_summary.items():
        cfg = r["config"]
        is_knn = "sim" in cfg
        ci = lambda k: "[%.4f, %.4f]" % tuple(r[k + "_ci"])
        notes = (f"Test set, {n_users} users evaluated. 95% bootstrap CI over users ({args['n_boot']} resamples): "
                 f"P@10 {ci('Precision@10')}; R@10 {ci('Recall@10')}; NDCG@10 {ci('NDCG@10')}. "
                 f"Coverage = distinct items in top-10 lists / {n_items} catalogue items. "
                 f"Setting selected on validation NDCG@10. Split: {args['split_mode']}, "
                 f"{args['train_frac']:.0%}/{args['val_frac']:.0%}/{1 - args['train_frac'] - args['val_frac']:.0%}; "
                 f"positive = rating >= {args['pos_threshold']}; ranked against full catalogue minus history.")
        rows.append({
            "Model": "Item-based kNN" if is_knn else "Most Popular",
            "Variant / settings": _variant_text(cfg),
            "Similarity": cfg["sim"] if is_knn else "n/a",
            "k": cfg["k"] if is_knn else "n/a",
            "Precision@10": round(r["Precision@10"], 6),
            "Recall@10": round(r["Recall@10"], 6),
            "NDCG@10": round(r["NDCG@10"], 6),
            "Catalogue coverage": round(r["Coverage@10"], 6),
            "Seed": args["seed"],
            "Notes": notes,
        })
    return rows


def write_template(test_summary, args, n_users, n_items, path):
    write_rows(template_rows(test_summary, args, n_users, n_items), path, TEMPLATE_COLUMNS)
