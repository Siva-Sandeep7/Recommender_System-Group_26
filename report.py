"""Generates outputs/report_numbers.md: every number and draft justification sentence
needed for the Part 1 report, filled in from the actual run."""


def _tbl(df, cols, fmt=None):
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt.get(c, lambda x: str(x))(v) if not isinstance(v, str) else v)
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def write_report_numbers(summary, tuning, sens, path):
    ds, sp = summary["data_stats"], summary["split"]
    fi, args = summary["filter_info"], summary["args"]
    te = summary["test"]
    f4 = lambda x: f"{x:.4f}"
    L = []
    a = L.append
    a("# Part 1 - numbers and draft text for the report\n")
    a("Paste these into the course report template (max 4 pages). Figures are in `figures/`.\n")

    a("## 1. Data characterization\n")
    a(f"- {ds['n_users']:,} users, {ds['n_items_rated']:,} rated items, {ds['n_ratings']:,} ratings; "
      f"density {100*ds['density']:.2f}%.")
    a(f"- Ratings per user: min {ds['ratings_per_user']['min']}, median {ds['ratings_per_user']['median']:.0f}, "
      f"mean {ds['ratings_per_user']['mean']:.1f}, max {ds['ratings_per_user']['max']}.")
    a(f"- Ratings per item: min {ds['ratings_per_item']['min']}, median {ds['ratings_per_item']['median']:.0f}, "
      f"mean {ds['ratings_per_item']['mean']:.1f}, max {ds['ratings_per_item']['max']}.")
    a(f"- Long tail: top 1% of items hold {100*ds['top1pct_items_share_of_ratings']:.1f}% of ratings, top 10% hold "
      f"{100*ds['top10pct_items_share_of_ratings']:.1f}%, top 20% hold {100*ds['top20pct_items_share_of_ratings']:.1f}%; "
      f"item-popularity Gini = {ds['item_popularity_gini']:.3f}; {ds['items_with_lt10_ratings_pct']:.1f}% of items have "
      f"fewer than 10 ratings ({ds['items_with_lt5_ratings_pct']:.1f}% fewer than 5).")
    a(f"- Rating distribution {ds['rating_counts']}; mean {ds['rating_mean']:.2f}; {ds['pct_ratings_ge4']:.1f}% of ratings are >= 4.")
    a(f"- Time span {ds['time_start']} to {ds['time_end']} ({ds['time_span_days']:.0f} days). Ratings per year: {ds['ratings_per_year']}.")
    a(f"- Burstiness: busiest day = {ds['busiest_day']} ({ds['busiest_day_share_pct']:.1f}% of all ratings); busiest 30 days = "
      f"{ds['busiest_30_days_share_pct']:.1f}%. {ds['users_all_ratings_within_1day_pct']:.1f}% of users entered ALL their ratings "
      f"within 24 h, {ds['users_all_ratings_within_7days_pct']:.1f}% within 7 days (median per-user span "
      f"{ds['user_rating_span_days_median']:.2f} days).")
    a("- Figures: `fig_long_tail.png`, `fig_rating_dist.png`, `fig_activity_time.png`.\n")

    a("## 2. Temporal split\n")
    if args.get("split_mode", "global") == "global":
        a(f"- Cut points (global, by timestamp quantile {args['train_frac']:.0%} / {args['train_frac']+args['val_frac']:.0%}): "
          f"train <= {sp['t_train_end']}, validation <= {sp['t_val_end']}, test after.")
    else:
        a(f"- Per-user chronological split: each user's earliest {args['train_frac']:.0%} of ratings are train, the next "
          f"{args['val_frac']:.0%} validation, the rest test (no single cut date; no cold-start users).")
    a(f"- Sizes: train {sp['n_train']:,} ({100*sp['frac_train']:.1f}%), validation {sp['n_val']:,} ({100*sp['frac_val']:.1f}%), "
      f"test {sp['n_test']:,} ({100*sp['frac_test']:.1f}%). Actual fractions differ slightly from the targets because many ratings share a timestamp.")
    for nm, key in [("Validation", "val_eval"), ("Test", "test_eval")]:
        e = sp[key]
        a(f"- {nm} evaluation: {e['n_eval_users']:,} users evaluated; {e['n_cold_users_excluded']:,} users with a positive "
          f"target item but no earlier history were excluded (cold-start users); "
          f"{e['pct_positive_target_on_items_unseen_in_fit']:.1f}% of positive target interactions are on items with no earlier interaction "
          f"(unreachable for CF but still counted as relevant).")
    a("")

    a("## 3. Filtering\n")
    a(f"- Main-run thresholds: min {fi['min_user']} ratings/user, min {fi['min_item']} ratings/item -> removed "
      f"{fi['rows_removed']:,} rows ({fi['pct_rows_removed']:.2f}%), {fi['users_removed']} users, {fi['items_removed']} items.")
    a("- Sensitivity (best baseline configs held fixed, item threshold varied):\n")
    a(_tbl(sens, ["min_item", "family", "rows_removed", "n_items", "n_eval_users", "Precision@10", "Recall@10", "NDCG@10", "Coverage@10"],
           {c: f4 for c in ["Precision@10", "Recall@10", "NDCG@10", "Coverage@10"]}))
    knn = sens[sens["family"] == "ItemKNN"]["NDCG@10"]
    pops = sens[sens["family"].str.startswith("MostPop")].groupby("min_item")["NDCG@10"].max()
    gap_main = float(knn.iloc[0] - pops.iloc[0])
    a(f"\n- Auto-check: kNN NDCG@10 ranges over {knn.min():.4f}-{knn.max():.4f} across thresholds "
      f"(spread {knn.max()-knn.min():.4f}); kNN minus best-MostPop gap at threshold {int(sens['min_item'].iloc[0])} is {gap_main:.4f}. "
      f"If the spread is larger than the gap, filtering matters more than the model choice - say so.\n")

    a("## 4. Tuning (validation only)\n")
    n = summary["n_tuning_configs"]
    a(f"- Budget: ItemKNN {n['knn']} configs (similarity x k); MostPop {n['pop']} configs across 4 variants "
      f"(window length, positive-only flag). Selection metric: NDCG@10 on validation.")
    a(f"- Best configs: {summary['best_configs']}\n")
    a(_tbl(tuning.sort_values(["family", "NDCG@10"], ascending=[True, False]),
           ["family", "config", "Precision@10", "Recall@10", "NDCG@10", "Coverage@10"],
           {c: f4 for c in ["Precision@10", "Recall@10", "NDCG@10", "Coverage@10"]}))

    a("\n## 5. Test results (refit on train+val; 95% bootstrap CI over users)\n")
    a("| Model | Config | P@10 | R@10 | NDCG@10 | Coverage@10 |\n|---|---|---|---|---|---|")
    for fam, r in te.items():
        ci = lambda k: f"{r[k]:.4f} [{r[k+'_ci'][0]:.4f}, {r[k+'_ci'][1]:.4f}]"
        a(f"| {fam} | {r['config']} | {ci('Precision@10')} | {ci('Recall@10')} | {ci('NDCG@10')} | {r['Coverage@10']:.4f} |")

    a("\n## 6. Draft protocol justifications (edit, 1-2 sentences each)\n")
    if args.get("split_mode", "global") == "global":
        a(f"- **Split**: a global temporal split ({args['train_frac']:.0%}/{args['val_frac']:.0%}/{1-args['train_frac']-args['val_frac']:.0%} "
          f"by timestamp) so that models never train on interactions that happen after the ones they are tested on (Lecture 3.1, slide 4); "
          f"cut points are listed above. Because timestamps are bursty, the cuts fall inside a very active period and split many users' "
          f"sessions - see the cold-user counts.")
    else:
        a(f"- **Split**: per-user chronological split ({args['train_frac']:.0%}/{args['val_frac']:.0%}/{1-args['train_frac']-args['val_frac']:.0%} "
          f"of each user's ratings in time order). A single global cut date leaves most users without earlier history because users rate "
          f"in short bursts (see burstiness above); the per-user split keeps every user warm. Limitation: no global cut date, so a test "
          f"interaction of one user can precede training interactions of another.")
    a(f"- **Feedback treatment**: ratings >= {args['pos_threshold']} are positive/relevant; all ratings are treated as 'seen' and "
      f"removed from the candidate list, so a disliked film is never recommended again. kNN is trained on positive feedback only.")
    a(f"- **Filtering**: main run uses min {fi['min_user']}/user and min {fi['min_item']}/item (removed {fi['pct_rows_removed']:.2f}% of rows); "
      f"the sensitivity table shows how much thresholds change the results relative to the baseline gap.")
    a("- **Candidate set**: the full catalogue of items that appear in the (filtered) data, minus the user's history; no sampled negatives, "
      "because sampled evaluation distorts rankings (state the lecture reference if covered).")
    a("- **Choice of k and similarity**: tuned on the validation split only, over the grid reported in section 4; final models refit on train+val and evaluated once on test.")
    a("- **Popularity variants**: all-time vs recent-window and all-ratings vs positive-only; window length tuned on validation.")
    if args.get("split_mode", "global") == "global":
        a("- **Cold users**: users with no history before the evaluation period cannot be personalised and are excluded and counted, not silently dropped.")
    a("- **Coverage**: number of distinct items appearing in any evaluated user's top-10 divided by the number of items in the catalogue.\n")

    a("## 7. You still need to add by hand\n")
    a("- GitHub repository URL; contribution statement (2-3 sentences per member); AI-use disclosure; seed (42 unless changed).")
    a("- Map `outputs/results.csv` onto the official results template columns (see `results.py`).")
    a("- Your own interpretation of the plots and of the filtering/burstiness findings above (this is where the 'excellent' marks are).")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
