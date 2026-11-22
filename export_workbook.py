"""Turns results_part2_*.csv into the layout of the course workbook tabs "Part 2 - Models" and
"Part 2 - Sparsity" (standard library only, so it runs without the virtual environment).

    python export_workbook.py outputs_part2_v2
"""
import csv
import sys
from pathlib import Path

STOCHASTIC = ("ALS", "BPR")
ORDER = ["Most Popular", "Item-based kNN", "User-based kNN", "ALS (implicit, library)",
         "BPR (self-implemented)", "EASE (self-implemented)"]


HYPER = {
    "Most Popular": lambda c: "counts: %s; window: %s" % ("positive ratings only" if c["positive_only"] else "all ratings",
                                                         "all-time" if c["window_days"] is None else "%d days" % c["window_days"]),
    "Item-based kNN": lambda c: "similarity=%s, k=%d neighbours per item" % (c["sim"], c["k"]),
    "User-based kNN": lambda c: "similarity=%s, k=%d neighbours, significance shrinkage=%d, IDF weighting=%s" % (
        c["sim"], c["k"], c["shrink"], "yes" if c["idf"] else "no"),
    "ALS (implicit, library)": lambda c: "factors=%d, L2 regularisation=%g, confidence alpha=%g, iterations=%d" % (
        c["factors"], c["reg"], c["alpha"], c["iterations"]),
    "BPR (self-implemented)": lambda c: "factors=%d, learning rate=%g, L2 regularisation=%g, epochs=%d, uniform negative sampling" % (
        c["k"], c["lr"], c["reg"], c["epochs"]),
    "EASE (self-implemented)": lambda c: "lambda (ridge penalty)=%g; closed-form solution, diag(B)=0" % c["lam"],
}
PARADIGM = {
    "Most Popular": "Non-personalised baseline (Part 1)",
    "Item-based kNN": "Neighbourhood CF, item-based (Part 1 baseline)",
    "User-based kNN": "Neighbourhood CF, user-based",
    "ALS (implicit, library)": "Latent-factor CF (weighted implicit-feedback matrix factorisation, ALS)",
    "BPR (self-implemented)": "Latent-factor CF with pairwise ranking loss (BPR-MF)",
    "EASE (self-implemented)": "Linear item-item autoencoder (closed-form ridge regression)",
}


def export_models(folder):
    import json
    res = {r["model"]: r for r in csv.DictReader(open(folder / "results_part2_models.csv", encoding="utf-8"))}
    pairs = {(r["model_a"], r["model_b"]): r for r in csv.DictReader(open(folder / "pairwise_ndcg.csv", encoding="utf-8"))}
    tune = {}
    for r in csv.DictReader(open(folder / "tuning_log_part2.csv", encoding="utf-8")):
        t = tune.setdefault(r["family"], [0, 0.0])
        t[0] += 1
        t[1] = max(t[1], float(r["NDCG@10"]))
    ref = "Item-based kNN"          # best Part 1 baseline
    out = []
    for m in ORDER:
        r = res[m]
        stoch = m.startswith(STOCHASTIC)
        var = "NDCG@10 95%% bootstrap CI over %s test users [%.4f, %.4f]" % (r["n_test_users"], float(r["NDCG@10 CI low"]), float(r["NDCG@10 CI high"]))
        if stoch:
            var += "; std over %s seeds: NDCG@10 %.4f, P@10 %.4f, R@10 %.4f" % (r["n_seeds"], float(r["NDCG@10 std over seeds"]),
                                                                            float(r["Precision@10 std over seeds"]), float(r["Recall@10 std over seeds"]))
        else:
            var += "; deterministic model, no seed variance"
        if m != ref and (m, ref) in pairs:
            p = pairs[(m, ref)]
            var += "; paired bootstrap vs %s: dNDCG@10 %+.4f [%.4f, %.4f]%s" % (
                ref, float(p["mean_diff_ndcg"]), float(p["ci_low"]), float(p["ci_high"]),
                " (CI excludes 0)" if p["ci_excludes_zero"] == "True" else " (CI includes 0)")
        n_cfg, best_val = tune[m]
        note = ("Final test results (%s test users) after refitting on train+validation; settings selected on validation NDCG@10 only "
                "(%d configurations tried, best validation NDCG@10 %.4f)%s. Coverage = distinct items in all top-10 lists / 3706 catalogue items. "
                "Per-user chronological 80/10/10 split, positive = rating >= 4, full-catalogue ranking excluding every rated item."
                % (r["n_test_users"], n_cfg, best_val, " - exhaustive grid" if n_cfg < 18 else ""))
        if m.startswith("EASE"):
            note += " Method: Steck, 'Embarrassingly Shallow Autoencoders for Sparse Data', WWW 2019; implemented from scratch with NumPy (ease.py)."
        if m.startswith("BPR"):
            note += " Written from scratch with NumPy (bpr.py); best settings lie at the edge of the searched grid (lr 0.1, 30 epochs), so it may be under-tuned."
        if m.startswith("ALS"):
            note += " Uses the 'implicit' library (not self-implemented)."
        out.append({"Model": m, "Paradigm": PARADIGM[m], "Self-implemented?": "Yes" if "self-implemented" in m else "No",
                    "Key hyperparameters": HYPER[m](json.loads(r["config"])),
                    "Precision@10": round(float(r["Precision@10"]), 6), "Recall@10": round(float(r["Recall@10"]), 6),
                    "NDCG@10": round(float(r["NDCG@10"]), 6), "Catalogue coverage": round(float(r["Coverage@10"]), 6),
                    "Variability / CI / test": var, "Seed(s)": "42-46 (mean)" if stoch else "n/a (deterministic)", "Notes": note})
    cols = list(out[0])
    with open(folder / "workbook_part2_models.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print("wrote", folder / "workbook_part2_models.csv", len(out), "rows")


def main(folder):
    folder = Path(folder)
    export_models(folder)
    rows = list(csv.DictReader(open(folder / "results_part2_sparsity.csv", encoding="utf-8")))
    n_models = len({r["model"] for r in rows})
    out = []
    for m in ORDER:
        for frac in ("1.0", "0.5", "0.2"):
            r = next((x for x in rows if x["model"] == m and x["fraction"] == frac), None)
            if r is None:
                continue
            stoch = m.startswith(STOCHASTIC)
            if frac == "1.0":
                seeds = "42-46 (mean)" if stoch else "n/a (deterministic)"
                how = "full training data (train+validation)"
            else:
                seeds = "42-46 (mean; seed = subsample%s)" % (" and model init" if stoch else "")
                how = "%d%% of the train+validation interactions kept, uniformly at random, a different subsample per seed" % round(100 * float(frac))
            note = ("%s. Mean of %s run(s); NDCG@10 std over runs %.4f; rank %s of %d by NDCG@10 at this fraction. "
                    "Hyper-parameters from the 100%% tuning kept fixed; interactions removed from training still count as seen "
                    "(excluded from the ranking); test users and test items unchanged."
                    % (how, r["n_runs"], float(r["ndcg_std_over_runs"]), r["rank_by_ndcg"], n_models))
            out.append({"Model": m, "Training interactions (%)": round(100 * float(frac)),
                        "Precision@10": round(float(r["Precision@10"]), 6), "Recall@10": round(float(r["Recall@10"]), 6),
                        "NDCG@10": round(float(r["NDCG@10"]), 6), "Catalogue coverage": round(float(r["Coverage@10"]), 6),
                        "Seed(s)": seeds, "Notes": note})
    cols = ["Model", "Training interactions (%)", "Precision@10", "Recall@10", "NDCG@10", "Catalogue coverage", "Seed(s)", "Notes"]
    with open(folder / "workbook_part2_sparsity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print("wrote", folder / "workbook_part2_sparsity.csv", len(out), "rows")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs_part2")
