# Part 2 - numbers for the report

## 1. Protocol recap

- Split: per_user; test users evaluated: 5,821; seeds: [42, 43, 44, 45, 46].
- Tuning budget: 18 validation configurations per model family (a family whose whole space is smaller is searched exhaustively); selection metric NDCG@10 on validation only; stochastic models tuned with seed 42.
- Disaggregation thresholds: low/high activity split at the median of 89 ratings in the fitting data (train+validation); popular items = the 741 most-rated items (top 20%, at least 380 ratings), all others long tail.

## 2. Tuning (validation)

| family | configs | best_val_NDCG | worst_val_NDCG |
|---|---|---|---|
| ALS (implicit, library) | 18 | 0.0872 | 0.0638 |
| BPR (self-implemented) | 18 | 0.0826 | 0.0577 |
| Item-based kNN | 18 | 0.0915 | 0.0741 |
| Most Popular | 12 | 0.0585 | 0.0303 |
| User-based kNN | 18 | 0.0916 | 0.0844 |

Best configurations: {"pop": {"positive_only": true, "window_days": null}, "iknn": {"sim": "cosine", "k": 200}, "uknn": {"sim": "jaccard", "idf": false, "k": 50, "shrink": 30}, "als": {"factors": 64, "reg": 1.0, "alpha": 5, "iterations": 15}, "bpr": {"k": 64, "lr": 0.1, "reg": 0.01, "epochs": 30}}

## 3. Overall test results (mean over seeds; 95% bootstrap CI over users; std over seeds)

| model | n_seeds | Precision@10 | Recall@10 | NDCG@10 | NDCG@10 CI low | NDCG@10 CI high | NDCG@10 std over seeds | Coverage@10 |
|---|---|---|---|---|---|---|---|---|
| Most Popular | 1 | 0.0455 | 0.0493 | 0.0584 | 0.0553 | 0.0616 | 0.0000 | 0.0391 |
| Item-based kNN | 1 | 0.0631 | 0.0850 | 0.0883 | 0.0842 | 0.0921 | 0.0000 | 0.2593 |
| User-based kNN | 1 | 0.0642 | 0.0887 | 0.0910 | 0.0867 | 0.0948 | 0.0000 | 0.1870 |
| ALS (implicit, library) | 5 | 0.0594 | 0.0922 | 0.0882 | 0.0845 | 0.0919 | 0.0007 | 0.3160 |
| BPR (self-implemented) | 5 | 0.0585 | 0.0817 | 0.0822 | 0.0790 | 0.0852 | 0.0007 | 0.3525 |

## 4. Paired NDCG@10 differences (model_a minus model_b; CI over users)

| model_a | model_b | mean_diff_ndcg | ci_low | ci_high | ci_excludes_zero |
|---|---|---|---|---|---|
| Most Popular | Item-based kNN | -0.0299 | -0.0334 | -0.0261 | True |
| Most Popular | User-based kNN | -0.0326 | -0.0361 | -0.0287 | True |
| Most Popular | ALS (implicit, library) | -0.0298 | -0.0339 | -0.0256 | True |
| Most Popular | BPR (self-implemented) | -0.0237 | -0.0273 | -0.0204 | True |
| Item-based kNN | Most Popular | 0.0299 | 0.0261 | 0.0334 | True |
| Item-based kNN | User-based kNN | -0.0027 | -0.0053 | 0.0001 | False |
| Item-based kNN | ALS (implicit, library) | 0.0001 | -0.0035 | 0.0034 | False |
| Item-based kNN | BPR (self-implemented) | 0.0062 | 0.0034 | 0.0086 | True |
| User-based kNN | Most Popular | 0.0326 | 0.0287 | 0.0361 | True |
| User-based kNN | Item-based kNN | 0.0027 | -0.0001 | 0.0053 | False |
| User-based kNN | ALS (implicit, library) | 0.0028 | -0.0004 | 0.0061 | False |
| User-based kNN | BPR (self-implemented) | 0.0089 | 0.0057 | 0.0120 | True |
| ALS (implicit, library) | Most Popular | 0.0298 | 0.0256 | 0.0339 | True |
| ALS (implicit, library) | Item-based kNN | -0.0001 | -0.0034 | 0.0035 | False |
| ALS (implicit, library) | User-based kNN | -0.0028 | -0.0061 | 0.0004 | False |
| ALS (implicit, library) | BPR (self-implemented) | 0.0061 | 0.0030 | 0.0090 | True |
| BPR (self-implemented) | Most Popular | 0.0237 | 0.0204 | 0.0273 | True |
| BPR (self-implemented) | Item-based kNN | -0.0062 | -0.0086 | -0.0034 | True |
| BPR (self-implemented) | User-based kNN | -0.0089 | -0.0120 | -0.0057 | True |
| BPR (self-implemented) | ALS (implicit, library) | -0.0061 | -0.0090 | -0.0030 | True |

## 5. Disaggregated results

| model | dimension | group | metric | value | ci_low | ci_high | n | slot_share |
|---|---|---|---|---|---|---|---|---|
| Most Popular | user activity | low activity (<= median) | NDCG@10 | 0.0389 | 0.0347 | 0.0428 | 2916 | nan |
| Most Popular | user activity | low activity (<= median) | Recall@10 | 0.0521 | 0.0470 | 0.0575 | 2916 | nan |
| Most Popular | user activity | high activity (> median) | NDCG@10 | 0.0780 | 0.0733 | 0.0831 | 2905 | nan |
| Most Popular | user activity | high activity (> median) | Recall@10 | 0.0465 | 0.0434 | 0.0498 | 2905 | nan |
| Most Popular | item popularity | popular items (head) | Recall@10 (relevant items of the group) | 0.0667 | 0.0629 | 0.0705 | 5480 | 1.0000 |
| Most Popular | item popularity | long-tail items | Recall@10 (relevant items of the group) | 0.0000 | 0.0000 | 0.0000 | 4251 | 0.0000 |
| Item-based kNN | user activity | low activity (<= median) | NDCG@10 | 0.0750 | 0.0694 | 0.0806 | 2916 | nan |
| Item-based kNN | user activity | low activity (<= median) | Recall@10 | 0.1076 | 0.1002 | 0.1156 | 2916 | nan |
| Item-based kNN | user activity | high activity (> median) | NDCG@10 | 0.1017 | 0.0964 | 0.1073 | 2905 | nan |
| Item-based kNN | user activity | high activity (> median) | Recall@10 | 0.0624 | 0.0586 | 0.0658 | 2905 | nan |
| Item-based kNN | item popularity | popular items (head) | Recall@10 (relevant items of the group) | 0.1168 | 0.1111 | 0.1223 | 5480 | 0.9808 |
| Item-based kNN | item popularity | long-tail items | Recall@10 (relevant items of the group) | 0.0045 | 0.0029 | 0.0063 | 4251 | 0.0192 |
| User-based kNN | user activity | low activity (<= median) | NDCG@10 | 0.0818 | 0.0758 | 0.0876 | 2916 | nan |
| User-based kNN | user activity | low activity (<= median) | Recall@10 | 0.1145 | 0.1065 | 0.1221 | 2916 | nan |
| User-based kNN | user activity | high activity (> median) | NDCG@10 | 0.1003 | 0.0948 | 0.1058 | 2905 | nan |
| User-based kNN | user activity | high activity (> median) | Recall@10 | 0.0628 | 0.0590 | 0.0664 | 2905 | nan |
| User-based kNN | item popularity | popular items (head) | Recall@10 (relevant items of the group) | 0.1229 | 0.1171 | 0.1284 | 5480 | 0.9939 |
| User-based kNN | item popularity | long-tail items | Recall@10 (relevant items of the group) | 0.0025 | 0.0013 | 0.0039 | 4251 | 0.0061 |
| ALS (implicit, library) | user activity | low activity (<= median) | NDCG@10 | 0.0866 | 0.0810 | 0.0923 | 2916 | nan |
| ALS (implicit, library) | user activity | low activity (<= median) | Recall@10 | 0.1245 | 0.1172 | 0.1320 | 2916 | nan |
| ALS (implicit, library) | user activity | high activity (> median) | NDCG@10 | 0.0899 | 0.0851 | 0.0944 | 2905 | nan |
| ALS (implicit, library) | user activity | high activity (> median) | Recall@10 | 0.0597 | 0.0561 | 0.0631 | 2905 | nan |
| ALS (implicit, library) | item popularity | popular items (head) | Recall@10 (relevant items of the group) | 0.1263 | 0.1205 | 0.1322 | 5480 | 0.9515 |
| ALS (implicit, library) | item popularity | long-tail items | Recall@10 (relevant items of the group) | 0.0093 | 0.0072 | 0.0114 | 4251 | 0.0485 |
| BPR (self-implemented) | user activity | low activity (<= median) | NDCG@10 | 0.0725 | 0.0685 | 0.0766 | 2916 | nan |
| BPR (self-implemented) | user activity | low activity (<= median) | Recall@10 | 0.1044 | 0.0989 | 0.1097 | 2916 | nan |
| BPR (self-implemented) | user activity | high activity (> median) | NDCG@10 | 0.0918 | 0.0874 | 0.0964 | 2905 | nan |
| BPR (self-implemented) | user activity | high activity (> median) | Recall@10 | 0.0589 | 0.0559 | 0.0620 | 2905 | nan |
| BPR (self-implemented) | item popularity | popular items (head) | Recall@10 (relevant items of the group) | 0.1105 | 0.1065 | 0.1147 | 5480 | 0.9351 |
| BPR (self-implemented) | item popularity | long-tail items | Recall@10 (relevant items of the group) | 0.0100 | 0.0085 | 0.0118 | 4251 | 0.0649 |

- Auto-check: best model for item popularity / long-tail items (Recall@10 (relevant items of the group)): **BPR (self-implemented)** (0.0100).
- Auto-check: best model for item popularity / popular items (head) (Recall@10 (relevant items of the group)): **ALS (implicit, library)** (0.1263).
- Auto-check: best model for user activity / high activity (> median) (NDCG@10): **Item-based kNN** (0.1017).
- Auto-check: best model for user activity / low activity (<= median) (Recall@10): **ALS (implicit, library)** (0.1245).
- Overall best on NDCG@10: **User-based kNN**. Any group above whose best model differs is a finding that qualifies the headline.

## 6. Sparsity ablation (mean NDCG@10 over runs; rank 1 = best)

| model | fraction | n_runs | Precision@10 | Recall@10 | NDCG@10 | Coverage@10 | ndcg_std_over_runs | rank_by_ndcg |
|---|---|---|---|---|---|---|---|---|
| Most Popular | 1.0000 | 1 | 0.0455 | 0.0493 | 0.0584 | 0.0391 | 0.0000 | 5 |
| Item-based kNN | 1.0000 | 1 | 0.0631 | 0.0850 | 0.0883 | 0.2593 | 0.0000 | 2 |
| User-based kNN | 1.0000 | 1 | 0.0642 | 0.0887 | 0.0910 | 0.1870 | 0.0000 | 1 |
| ALS (implicit, library) | 1.0000 | 5 | 0.0594 | 0.0922 | 0.0882 | 0.3160 | 0.0007 | 3 |
| BPR (self-implemented) | 1.0000 | 5 | 0.0585 | 0.0817 | 0.0822 | 0.3525 | 0.0007 | 4 |
| Most Popular | 0.5000 | 5 | 0.0456 | 0.0503 | 0.0589 | 0.0387 | 0.0004 | 5 |
| Item-based kNN | 0.5000 | 5 | 0.0583 | 0.0752 | 0.0803 | 0.3410 | 0.0010 | 1 |
| User-based kNN | 0.5000 | 5 | 0.0539 | 0.0717 | 0.0754 | 0.1900 | 0.0020 | 2 |
| ALS (implicit, library) | 0.5000 | 5 | 0.0480 | 0.0703 | 0.0695 | 0.2588 | 0.0013 | 4 |
| BPR (self-implemented) | 0.5000 | 5 | 0.0506 | 0.0680 | 0.0698 | 0.3561 | 0.0018 | 3 |
| Most Popular | 0.2000 | 5 | 0.0454 | 0.0495 | 0.0582 | 0.0389 | 0.0007 | 1 |
| Item-based kNN | 0.2000 | 5 | 0.0347 | 0.0408 | 0.0453 | 0.7202 | 0.0008 | 5 |
| User-based kNN | 0.2000 | 5 | 0.0385 | 0.0465 | 0.0517 | 0.3217 | 0.0013 | 3 |
| ALS (implicit, library) | 0.2000 | 5 | 0.0346 | 0.0443 | 0.0472 | 0.2161 | 0.0004 | 4 |
| BPR (self-implemented) | 0.2000 | 5 | 0.0404 | 0.0500 | 0.0535 | 0.3056 | 0.0010 | 2 |

- 100% of training interactions, best to worst: User-based kNN > Item-based kNN > ALS (implicit, library) > BPR (self-implemented) > Most Popular
- 50% of training interactions, best to worst: Item-based kNN > User-based kNN > BPR (self-implemented) > ALS (implicit, library) > Most Popular
- 20% of training interactions, best to worst: Most Popular > BPR (self-implemented) > User-based kNN > ALS (implicit, library) > Item-based kNN
- Ranking changes between 100% and 20%: **yes**.

## 7. Still to write by hand

- Interpretation: why the ranking looks the way it does overall, per group and under sparsity; which differences have a CI that excludes zero.
- Contribution, AI-use disclosure, Part 3 track announcement, repository URL.
