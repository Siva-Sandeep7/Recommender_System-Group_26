# Part 2 - numbers for the report

## 1. Protocol recap

- Split: per_user; test users evaluated: 5,821; seeds: [42, 43].
- Tuning budget: 18 validation configurations per model family (a family whose whole space is smaller is searched exhaustively); selection metric NDCG@10 on validation only; stochastic models tuned with seed 42.
- Disaggregation thresholds: low/high activity split at the median of 89 ratings in the fitting data (train+validation); popular items = the 741 most-rated items (top 20%, at least 380 ratings), all others long tail.

## 2. Tuning (validation)

| family | configs | best_val_NDCG | worst_val_NDCG |
|---|---|---|---|
| ALS (implicit, library) | 4 | 0.0855 | 0.0717 |
| BPR (self-implemented) | 4 | 0.0632 | 0.0571 |
| Item-based kNN | 2 | 0.0885 | 0.0840 |
| Most Popular | 6 | 0.0585 | 0.0303 |
| User-based kNN | 12 | 0.0916 | 0.0844 |

Best configurations: {"pop": {"positive_only": true, "window_days": null}, "iknn": {"sim": "cosine", "k": 50}, "uknn": {"sim": "jaccard", "idf": false, "k": 50, "shrink": 30}, "als": {"factors": 16, "reg": 0.1, "alpha": 5, "iterations": 5}, "bpr": {"k": 16, "lr": 0.05, "reg": 0.01, "epochs": 4}}

## 3. Overall test results (mean over seeds; 95% bootstrap CI over users; std over seeds)

| model | n_seeds | Precision@10 | Recall@10 | NDCG@10 | NDCG@10 CI low | NDCG@10 CI high | NDCG@10 std over seeds | Coverage@10 |
|---|---|---|---|---|---|---|---|---|
| Most Popular | 1 | 0.0455 | 0.0493 | 0.0584 | 0.0553 | 0.0616 | 0.0000 | 0.0391 |
| Item-based kNN | 1 | 0.0616 | 0.0861 | 0.0870 | 0.0831 | 0.0908 | 0.0000 | 0.3049 |
| User-based kNN | 1 | 0.0642 | 0.0887 | 0.0910 | 0.0867 | 0.0948 | 0.0000 | 0.1870 |
| ALS (implicit, library) | 2 | 0.0607 | 0.0860 | 0.0869 | 0.0833 | 0.0903 | 0.0012 | 0.2159 |
| BPR (self-implemented) | 2 | 0.0475 | 0.0493 | 0.0605 | 0.0575 | 0.0634 | 0.0013 | 0.0739 |

## 4. Paired NDCG@10 differences (model_a minus model_b; CI over users)

| model_a | model_b | mean_diff_ndcg | ci_low | ci_high | ci_excludes_zero |
|---|---|---|---|---|---|
| Most Popular | Item-based kNN | -0.0286 | -0.0322 | -0.0248 | True |
| Most Popular | User-based kNN | -0.0326 | -0.0361 | -0.0287 | True |
| Most Popular | ALS (implicit, library) | -0.0285 | -0.0322 | -0.0247 | True |
| Most Popular | BPR (self-implemented) | -0.0021 | -0.0036 | -0.0004 | True |
| Item-based kNN | Most Popular | 0.0286 | 0.0248 | 0.0322 | True |
| Item-based kNN | User-based kNN | -0.0040 | -0.0071 | -0.0012 | True |
| Item-based kNN | ALS (implicit, library) | 0.0001 | -0.0029 | 0.0030 | False |
| Item-based kNN | BPR (self-implemented) | 0.0266 | 0.0229 | 0.0302 | True |
| User-based kNN | Most Popular | 0.0326 | 0.0287 | 0.0361 | True |
| User-based kNN | Item-based kNN | 0.0040 | 0.0012 | 0.0071 | True |
| User-based kNN | ALS (implicit, library) | 0.0041 | 0.0011 | 0.0070 | True |
| User-based kNN | BPR (self-implemented) | 0.0306 | 0.0270 | 0.0338 | True |
| ALS (implicit, library) | Most Popular | 0.0285 | 0.0247 | 0.0322 | True |
| ALS (implicit, library) | Item-based kNN | -0.0001 | -0.0030 | 0.0029 | False |
| ALS (implicit, library) | User-based kNN | -0.0041 | -0.0070 | -0.0011 | True |
| ALS (implicit, library) | BPR (self-implemented) | 0.0264 | 0.0229 | 0.0300 | True |
| BPR (self-implemented) | Most Popular | 0.0021 | 0.0004 | 0.0036 | True |
| BPR (self-implemented) | Item-based kNN | -0.0266 | -0.0302 | -0.0229 | True |
| BPR (self-implemented) | User-based kNN | -0.0306 | -0.0338 | -0.0270 | True |
| BPR (self-implemented) | ALS (implicit, library) | -0.0264 | -0.0300 | -0.0229 | True |

## 5. Disaggregated results

| model | dimension | group | metric | value | ci_low | ci_high | n | slot_share |
|---|---|---|---|---|---|---|---|---|
| Most Popular | user activity | low activity (<= median) | NDCG@10 | 0.0389 | 0.0347 | 0.0428 | 2916 | nan |
| Most Popular | user activity | low activity (<= median) | Recall@10 | 0.0521 | 0.0470 | 0.0575 | 2916 | nan |
| Most Popular | user activity | high activity (> median) | NDCG@10 | 0.0780 | 0.0733 | 0.0831 | 2905 | nan |
| Most Popular | user activity | high activity (> median) | Recall@10 | 0.0465 | 0.0434 | 0.0498 | 2905 | nan |
| Most Popular | item popularity | popular items (head) | Recall@10 (relevant items of the group) | 0.0667 | 0.0629 | 0.0705 | 5480 | 1.0000 |
| Most Popular | item popularity | long-tail items | Recall@10 (relevant items of the group) | 0.0000 | 0.0000 | 0.0000 | 4251 | 0.0000 |
| Item-based kNN | user activity | low activity (<= median) | NDCG@10 | 0.0780 | 0.0723 | 0.0834 | 2916 | nan |
| Item-based kNN | user activity | low activity (<= median) | Recall@10 | 0.1108 | 0.1031 | 0.1181 | 2916 | nan |
| Item-based kNN | user activity | high activity (> median) | NDCG@10 | 0.0961 | 0.0911 | 0.1010 | 2905 | nan |
| Item-based kNN | user activity | high activity (> median) | Recall@10 | 0.0614 | 0.0575 | 0.0648 | 2905 | nan |
| Item-based kNN | item popularity | popular items (head) | Recall@10 (relevant items of the group) | 0.1165 | 0.1106 | 0.1222 | 5480 | 0.9637 |
| Item-based kNN | item popularity | long-tail items | Recall@10 (relevant items of the group) | 0.0087 | 0.0064 | 0.0109 | 4251 | 0.0363 |
| User-based kNN | user activity | low activity (<= median) | NDCG@10 | 0.0818 | 0.0758 | 0.0876 | 2916 | nan |
| User-based kNN | user activity | low activity (<= median) | Recall@10 | 0.1145 | 0.1065 | 0.1221 | 2916 | nan |
| User-based kNN | user activity | high activity (> median) | NDCG@10 | 0.1003 | 0.0948 | 0.1058 | 2905 | nan |
| User-based kNN | user activity | high activity (> median) | Recall@10 | 0.0628 | 0.0590 | 0.0664 | 2905 | nan |
| User-based kNN | item popularity | popular items (head) | Recall@10 (relevant items of the group) | 0.1229 | 0.1171 | 0.1284 | 5480 | 0.9939 |
| User-based kNN | item popularity | long-tail items | Recall@10 (relevant items of the group) | 0.0025 | 0.0013 | 0.0039 | 4251 | 0.0061 |
| ALS (implicit, library) | user activity | low activity (<= median) | NDCG@10 | 0.0762 | 0.0712 | 0.0816 | 2916 | nan |
| ALS (implicit, library) | user activity | low activity (<= median) | Recall@10 | 0.1099 | 0.1026 | 0.1168 | 2916 | nan |
| ALS (implicit, library) | user activity | high activity (> median) | NDCG@10 | 0.0976 | 0.0927 | 0.1024 | 2905 | nan |
| ALS (implicit, library) | user activity | high activity (> median) | Recall@10 | 0.0620 | 0.0583 | 0.0654 | 2905 | nan |
| ALS (implicit, library) | item popularity | popular items (head) | Recall@10 (relevant items of the group) | 0.1202 | 0.1148 | 0.1254 | 5480 | 0.9878 |
| ALS (implicit, library) | item popularity | long-tail items | Recall@10 (relevant items of the group) | 0.0019 | 0.0011 | 0.0028 | 4251 | 0.0122 |
| BPR (self-implemented) | user activity | low activity (<= median) | NDCG@10 | 0.0374 | 0.0336 | 0.0408 | 2916 | nan |
| BPR (self-implemented) | user activity | low activity (<= median) | Recall@10 | 0.0498 | 0.0451 | 0.0541 | 2916 | nan |
| BPR (self-implemented) | user activity | high activity (> median) | NDCG@10 | 0.0836 | 0.0792 | 0.0887 | 2905 | nan |
| BPR (self-implemented) | user activity | high activity (> median) | Recall@10 | 0.0488 | 0.0460 | 0.0519 | 2905 | nan |
| BPR (self-implemented) | item popularity | popular items (head) | Recall@10 (relevant items of the group) | 0.0683 | 0.0648 | 0.0717 | 5480 | 0.9999 |
| BPR (self-implemented) | item popularity | long-tail items | Recall@10 (relevant items of the group) | 0.0000 | 0.0000 | 0.0000 | 4251 | 0.0001 |

- Auto-check: best model for item popularity / long-tail items (Recall@10 (relevant items of the group)): **Item-based kNN** (0.0087).
- Auto-check: best model for item popularity / popular items (head) (Recall@10 (relevant items of the group)): **User-based kNN** (0.1229).
- Auto-check: best model for user activity / high activity (> median) (NDCG@10): **User-based kNN** (0.1003).
- Auto-check: best model for user activity / low activity (<= median) (Recall@10): **User-based kNN** (0.1145).
- Overall best on NDCG@10: **User-based kNN**. Any group above whose best model differs is a finding that qualifies the headline.

## 6. Sparsity ablation (mean NDCG@10 over runs; rank 1 = best)

| model | fraction | n_runs | Precision@10 | Recall@10 | NDCG@10 | Coverage@10 | ndcg_std_over_runs | rank_by_ndcg |
|---|---|---|---|---|---|---|---|---|
| Most Popular | 1.0000 | 1 | 0.0455 | 0.0493 | 0.0584 | 0.0391 | 0.0000 | 5 |
| Item-based kNN | 1.0000 | 1 | 0.0616 | 0.0861 | 0.0870 | 0.3049 | 0.0000 | 2 |
| User-based kNN | 1.0000 | 1 | 0.0642 | 0.0887 | 0.0910 | 0.1870 | 0.0000 | 1 |
| ALS (implicit, library) | 1.0000 | 2 | 0.0607 | 0.0860 | 0.0869 | 0.2159 | 0.0012 | 3 |
| BPR (self-implemented) | 1.0000 | 2 | 0.0475 | 0.0493 | 0.0605 | 0.0739 | 0.0013 | 4 |
| Most Popular | 0.5000 | 2 | 0.0457 | 0.0507 | 0.0592 | 0.0387 | 0.0003 | 4 |
| Item-based kNN | 0.5000 | 2 | 0.0548 | 0.0728 | 0.0757 | 0.4085 | 0.0019 | 3 |
| User-based kNN | 0.5000 | 2 | 0.0549 | 0.0734 | 0.0772 | 0.1894 | 0.0019 | 2 |
| ALS (implicit, library) | 0.5000 | 2 | 0.0571 | 0.0759 | 0.0797 | 0.1838 | 0.0026 | 1 |
| BPR (self-implemented) | 0.5000 | 2 | 0.0428 | 0.0463 | 0.0533 | 0.0397 | 0.0017 | 5 |
| Most Popular | 0.2000 | 2 | 0.0455 | 0.0506 | 0.0586 | 0.0380 | 0.0002 | 1 |
| Item-based kNN | 0.2000 | 2 | 0.0246 | 0.0323 | 0.0333 | 0.7029 | 0.0004 | 5 |
| User-based kNN | 0.2000 | 2 | 0.0382 | 0.0464 | 0.0514 | 0.3210 | 0.0003 | 4 |
| ALS (implicit, library) | 0.2000 | 2 | 0.0438 | 0.0516 | 0.0572 | 0.1469 | 0.0020 | 2 |
| BPR (self-implemented) | 0.2000 | 2 | 0.0427 | 0.0456 | 0.0534 | 0.0401 | 0.0005 | 3 |

- 100% of training interactions, best to worst: User-based kNN > Item-based kNN > ALS (implicit, library) > BPR (self-implemented) > Most Popular
- 50% of training interactions, best to worst: ALS (implicit, library) > User-based kNN > Item-based kNN > Most Popular > BPR (self-implemented)
- 20% of training interactions, best to worst: Most Popular > ALS (implicit, library) > BPR (self-implemented) > User-based kNN > Item-based kNN
- Ranking changes between 100% and 20%: **yes**.

## 7. Still to write by hand

- Interpretation: why the ranking looks the way it does overall, per group and under sparsity; which differences have a CI that excludes zero.
- Contribution, AI-use disclosure, Part 3 track announcement, repository URL.
