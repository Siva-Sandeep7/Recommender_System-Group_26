# Part 1 - numbers and draft text for the report

Paste these into the course report template (max 4 pages). Figures are in `figures/`.

## 1. Data characterization

- 6,040 users, 3,706 rated items, 1,000,209 ratings; density 4.47%.
- Ratings per user: min 20, median 96, mean 165.6, max 2314.
- Ratings per item: min 1, median 124, mean 269.9, max 3428.
- Long tail: top 1% of items hold 8.7% of ratings, top 10% hold 44.4%, top 20% hold 65.2%; item-popularity Gini = 0.634; 12.0% of items have fewer than 10 ratings (7.8% fewer than 5).
- Rating distribution {1: 56174, 2: 107557, 3: 261197, 4: 348971, 5: 226310}; mean 3.58; 57.5% of ratings are >= 4.
- Time span 2000-04-25 23:05:32 to 2003-02-28 17:49:50 (1039 days). Ratings per year: {2000: 904757, 2001: 68058, 2002: 24046, 2003: 3348}.
- Burstiness: busiest day = 2000-11-20 (6.4% of all ratings); busiest 30 days = 39.7%. 62.9% of users entered ALL their ratings within 24 h, 70.2% within 7 days (median per-user span 0.05 days).
- Figures: `fig_long_tail.png`, `fig_rating_dist.png`, `fig_activity_time.png`.

## 2. Temporal split

- Per-user chronological split: each user's earliest 80% of ratings are train, the next 10% validation, the rest test (no single cut date; no cold-start users).
- Sizes: train 797,758 (79.8%), validation 99,692 (10.0%), test 102,759 (10.3%). Actual fractions differ slightly from the targets because many ratings share a timestamp.
- Validation evaluation: 5,789 users evaluated; 0 users with a positive target item but no earlier history were excluded (cold-start users); 0.0% of positive target interactions are on items with no earlier interaction (unreachable for CF but still counted as relevant).
- Test evaluation: 5,821 users evaluated; 0 users with a positive target item but no earlier history were excluded (cold-start users); 0.0% of positive target interactions are on items with no earlier interaction (unreachable for CF but still counted as relevant).

## 3. Filtering

- Main-run thresholds: min 0 ratings/user, min 0 ratings/item -> removed 0 rows (0.00%), 0 users, 0 items.
- Sensitivity (best baseline configs held fixed, item threshold varied):

| min_item | family | rows_removed | n_items | n_eval_users | Precision@10 | Recall@10 | NDCG@10 | Coverage@10 |
|---|---|---|---|---|---|---|---|---|
| 0 | MostPop (all-time, all ratings) | 0 | 3706 | 5821 | 0.0445 | 0.0479 | 0.0569 | 0.0402 |
| 0 | MostPop (all-time, positive only) | 0 | 3706 | 5821 | 0.0455 | 0.0493 | 0.0584 | 0.0391 |
| 0 | MostPop (recent window, all ratings) | 0 | 3706 | 5821 | 0.0370 | 0.0432 | 0.0465 | 0.0305 |
| 0 | MostPop (recent window, positive only) | 0 | 3706 | 5821 | 0.0380 | 0.0387 | 0.0465 | 0.0291 |
| 0 | ItemKNN | 0 | 3706 | 5821 | 0.0631 | 0.0850 | 0.0883 | 0.2593 |
| 5 | MostPop (all-time, all ratings) | 598 | 3416 | 5820 | 0.0445 | 0.0480 | 0.0570 | 0.0433 |
| 5 | MostPop (all-time, positive only) | 598 | 3416 | 5820 | 0.0455 | 0.0493 | 0.0584 | 0.0424 |
| 5 | MostPop (recent window, all ratings) | 598 | 3416 | 5820 | 0.0370 | 0.0432 | 0.0465 | 0.0331 |
| 5 | MostPop (recent window, positive only) | 598 | 3416 | 5820 | 0.0380 | 0.0387 | 0.0465 | 0.0316 |
| 5 | ItemKNN | 598 | 3416 | 5820 | 0.0631 | 0.0851 | 0.0883 | 0.2796 |
| 10 | MostPop (all-time, all ratings) | 1670 | 3260 | 5819 | 0.0445 | 0.0480 | 0.0570 | 0.0457 |
| 10 | MostPop (all-time, positive only) | 1670 | 3260 | 5819 | 0.0456 | 0.0494 | 0.0585 | 0.0445 |
| 10 | MostPop (recent window, all ratings) | 1670 | 3260 | 5819 | 0.0370 | 0.0431 | 0.0464 | 0.0347 |
| 10 | MostPop (recent window, positive only) | 1670 | 3260 | 5819 | 0.0380 | 0.0387 | 0.0465 | 0.0331 |
| 10 | ItemKNN | 1670 | 3260 | 5819 | 0.0629 | 0.0850 | 0.0881 | 0.2939 |
| 20 | MostPop (all-time, all ratings) | 4717 | 3043 | 5819 | 0.0446 | 0.0480 | 0.0571 | 0.0486 |
| 20 | MostPop (all-time, positive only) | 4717 | 3043 | 5819 | 0.0456 | 0.0493 | 0.0585 | 0.0473 |
| 20 | MostPop (recent window, all ratings) | 4717 | 3043 | 5819 | 0.0362 | 0.0415 | 0.0450 | 0.0368 |
| 20 | MostPop (recent window, positive only) | 4717 | 3043 | 5819 | 0.0381 | 0.0387 | 0.0465 | 0.0352 |
| 20 | ItemKNN | 4717 | 3043 | 5819 | 0.0630 | 0.0851 | 0.0881 | 0.3155 |


- Auto-check: kNN NDCG@10 ranges over 0.0881-0.0883 across thresholds (spread 0.0002); kNN minus best-MostPop gap at threshold 0 is 0.0299. If the spread is larger than the gap, filtering matters more than the model choice - say so.

## 4. Tuning (validation only)

- Budget: ItemKNN 18 configs (similarity x k); MostPop 12 configs across 4 variants (window length, positive-only flag). Selection metric: NDCG@10 on validation.
- Best configs: {'MostPop (all-time, all ratings)': {'positive_only': False, 'window_days': None}, 'MostPop (all-time, positive only)': {'positive_only': True, 'window_days': None}, 'MostPop (recent window, all ratings)': {'positive_only': False, 'window_days': 180}, 'MostPop (recent window, positive only)': {'positive_only': True, 'window_days': 365}, 'ItemKNN': {'sim': 'cosine', 'k': 200}}

| family | config | Precision@10 | Recall@10 | NDCG@10 | Coverage@10 |
|---|---|---|---|---|---|
| ItemKNN | {"sim": "cosine", "k": 200} | 0.0627 | 0.0954 | 0.0915 | 0.2572 |
| ItemKNN | {"sim": "cosine", "k": 500} | 0.0619 | 0.0896 | 0.0910 | 0.1943 |
| ItemKNN | {"sim": "jaccard", "k": 100} | 0.0618 | 0.0939 | 0.0897 | 0.2415 |
| ItemKNN | {"sim": "cosine", "k": 100} | 0.0612 | 0.0957 | 0.0895 | 0.2893 |
| ItemKNN | {"sim": "jaccard", "k": 50} | 0.0608 | 0.0918 | 0.0891 | 0.2563 |
| ItemKNN | {"sim": "jaccard", "k": 200} | 0.0615 | 0.0898 | 0.0887 | 0.2199 |
| ItemKNN | {"sim": "jaccard", "k": 500} | 0.0608 | 0.0878 | 0.0885 | 0.1835 |
| ItemKNN | {"sim": "cosine", "k": 50} | 0.0606 | 0.0946 | 0.0885 | 0.3119 |
| ItemKNN | {"sim": "jaccard", "k": 20} | 0.0590 | 0.0905 | 0.0857 | 0.2701 |
| ItemKNN | {"sim": "cosine", "k": 20} | 0.0570 | 0.0903 | 0.0840 | 0.3278 |
| ItemKNN | {"sim": "cosine", "k": 10} | 0.0557 | 0.0894 | 0.0835 | 0.3413 |
| ItemKNN | {"sim": "jaccard", "k": 10} | 0.0567 | 0.0864 | 0.0828 | 0.2790 |
| ItemKNN | {"sim": "adjcos", "k": 50} | 0.0582 | 0.0793 | 0.0815 | 0.3041 |
| ItemKNN | {"sim": "adjcos", "k": 20} | 0.0570 | 0.0798 | 0.0802 | 0.3305 |
| ItemKNN | {"sim": "adjcos", "k": 100} | 0.0574 | 0.0776 | 0.0795 | 0.2795 |
| ItemKNN | {"sim": "adjcos", "k": 10} | 0.0540 | 0.0757 | 0.0773 | 0.3591 |
| ItemKNN | {"sim": "adjcos", "k": 200} | 0.0555 | 0.0729 | 0.0761 | 0.2493 |
| ItemKNN | {"sim": "adjcos", "k": 500} | 0.0546 | 0.0702 | 0.0741 | 0.2205 |
| MostPop (all-time, all ratings) | {"positive_only": false, "window_days": null} | 0.0435 | 0.0482 | 0.0568 | 0.0321 |
| MostPop (all-time, positive only) | {"positive_only": true, "window_days": null} | 0.0447 | 0.0493 | 0.0585 | 0.0335 |
| MostPop (recent window, all ratings) | {"positive_only": false, "window_days": 180} | 0.0349 | 0.0432 | 0.0472 | 0.0246 |
| MostPop (recent window, all ratings) | {"positive_only": false, "window_days": 90} | 0.0300 | 0.0363 | 0.0416 | 0.0221 |
| MostPop (recent window, all ratings) | {"positive_only": false, "window_days": 365} | 0.0314 | 0.0397 | 0.0398 | 0.0254 |
| MostPop (recent window, all ratings) | {"positive_only": false, "window_days": 7} | 0.0287 | 0.0337 | 0.0378 | 0.0286 |
| MostPop (recent window, all ratings) | {"positive_only": false, "window_days": 30} | 0.0200 | 0.0253 | 0.0303 | 0.0219 |
| MostPop (recent window, positive only) | {"positive_only": true, "window_days": 365} | 0.0373 | 0.0462 | 0.0462 | 0.0246 |
| MostPop (recent window, positive only) | {"positive_only": true, "window_days": 180} | 0.0337 | 0.0396 | 0.0454 | 0.0251 |
| MostPop (recent window, positive only) | {"positive_only": true, "window_days": 90} | 0.0321 | 0.0377 | 0.0438 | 0.0210 |
| MostPop (recent window, positive only) | {"positive_only": true, "window_days": 30} | 0.0313 | 0.0403 | 0.0397 | 0.0202 |
| MostPop (recent window, positive only) | {"positive_only": true, "window_days": 7} | 0.0298 | 0.0302 | 0.0360 | 0.0313 |


## 5. Test results (refit on train+val; 95% bootstrap CI over users)

| Model | Config | P@10 | R@10 | NDCG@10 | Coverage@10 |
|---|---|---|---|---|---|
| MostPop (all-time, all ratings) | {'positive_only': False, 'window_days': None} | 0.0445 [0.0422, 0.0467] | 0.0479 [0.0452, 0.0507] | 0.0569 [0.0539, 0.0601] | 0.0402 |
| MostPop (all-time, positive only) | {'positive_only': True, 'window_days': None} | 0.0455 [0.0431, 0.0479] | 0.0493 [0.0465, 0.0525] | 0.0584 [0.0553, 0.0616] | 0.0391 |
| MostPop (recent window, all ratings) | {'positive_only': False, 'window_days': 180} | 0.0370 [0.0353, 0.0388] | 0.0432 [0.0407, 0.0458] | 0.0465 [0.0441, 0.0489] | 0.0305 |
| MostPop (recent window, positive only) | {'positive_only': True, 'window_days': 365} | 0.0380 [0.0360, 0.0400] | 0.0387 [0.0364, 0.0411] | 0.0465 [0.0439, 0.0490] | 0.0291 |
| ItemKNN | {'sim': 'cosine', 'k': 200} | 0.0631 [0.0604, 0.0657] | 0.0850 [0.0805, 0.0892] | 0.0883 [0.0842, 0.0921] | 0.2593 |

## 6. Draft protocol justifications (edit, 1-2 sentences each)

- **Split**: per-user chronological split (80%/10%/10% of each user's ratings in time order). A single global cut date leaves most users without earlier history because users rate in short bursts (see burstiness above); the per-user split keeps every user warm. Limitation: no global cut date, so a test interaction of one user can precede training interactions of another.
- **Feedback treatment**: ratings >= 4 are positive/relevant; all ratings are treated as 'seen' and removed from the candidate list, so a disliked film is never recommended again. kNN is trained on positive feedback only.
- **Filtering**: main run uses min 0/user and min 0/item (removed 0.00% of rows); the sensitivity table shows how much thresholds change the results relative to the baseline gap.
- **Candidate set**: the full catalogue of items that appear in the (filtered) data, minus the user's history; no sampled negatives, because sampled evaluation distorts rankings (state the lecture reference if covered).
- **Choice of k and similarity**: tuned on the validation split only, over the grid reported in section 4; final models refit on train+val and evaluated once on test.
- **Popularity variants**: all-time vs recent-window and all-ratings vs positive-only; window length tuned on validation.
- **Coverage**: number of distinct items appearing in any evaluated user's top-10 divided by the number of items in the catalogue.

## 7. You still need to add by hand

- GitHub repository URL; contribution statement (2-3 sentences per member); AI-use disclosure; seed (42 unless changed).
- Map `outputs/results.csv` onto the official results template columns (see `results.py`).
- Your own interpretation of the plots and of the filtering/burstiness findings above (this is where the 'excellent' marks are).
