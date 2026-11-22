## Part 2 - Models and honest comparison

Adds three models to the Part 1 protocol (same data, same per-user chronological split, same metrics, same full-catalogue ranking) and compares them with the Part 1 baselines.

| Model | Paradigm | Source |
|---|---|---|
| Most Popular, Item-based kNN | baselines from Part 1, re-tuned here with the same budget | `models.py` |
| User-based kNN | neighbourhood CF | `models_p2.py` (NumPy/SciPy, no recommender library) |
| ALS | latent-factor CF (weighted implicit-feedback MF) | library `implicit` (wrapper in `models_p2.py`) |
| **BPR** | ranking-loss CF | **self-implemented** in `bpr.py` (NumPy only) |

### Run
```
pip install -r requirements.txt          # includes implicit
python run_part2.py --data-dir /path/to/ml-1m
```
Writes to `outputs_part2/` (`--out-dir`). Takes roughly 25-35 minutes on a laptop. Useful options: `--quick` (tiny smoke test), `--reuse-tuning` (skip tuning, reuse `best_configs.json`), `--skip-als`, `--n-seeds`, `--budget`.

| File | Content |
|---|---|
| `results_part2_models.csv` | overall test results, mean over seeds, bootstrap CIs, std over seeds |
| `pairwise_ndcg.csv` | paired per-user NDCG@10 differences with 95% CIs |
| `results_part2_disaggregated.csv` | results by user activity (low/high) and item popularity (head/long tail) |
| `results_part2_sparsity.csv` | 100% / 50% / 20% training interactions, with rank per fraction |
| `tuning_log_part2.csv`, `best_configs.json` | every validation run and the selected configurations |
| `report_numbers_part2.md` | all numbers used in the report |

### Protocol additions
- **Tuning:** validation only, the same budget (18 configurations) for every model family; a family with a smaller search space is searched exhaustively. Stochastic models are tuned with seed 42.
- **Variability:** stochastic models (ALS, BPR) are run with 5 seeds (42-46); intervals are 95% bootstrap CIs over users on the seed-averaged per-user metric, plus the standard deviation across seeds. Comparisons between models are paired (per-user differences).
- **Disaggregation:** low/high activity users split at the median number of ratings; popular items are the 20% most-rated items.
- **Sparsity:** a random fraction of the training interactions is kept (different subsample per seed). Hyperparameters chosen at 100% are held fixed. The candidate set stays the same: every interaction in the full history is still excluded from the ranking.

### Checks
```
python tests/check_bpr.py          # gradient check, hand-computed update, negative sampling, learning
python tests/check_models_p2.py    # user-kNN vs a loop implementation; ALS wrapper (if implicit is installed)
python tests/check_analysis.py     # disaggregation / paired-CI helpers vs brute force
```

### Self-implemented algorithm
BPR-MF (`bpr.py`, Lecture 2.1 slide 12): matrix factorisation with an item bias, trained with mini-batch stochastic gradient ascent on `ln sigma(x_uij) - reg/2 |params|^2`, with one uniformly sampled untouched item as negative per positive. The loss, gradients and update rule are documented at the top of the file.
