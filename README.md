# Recommender Systems course project - Part 1 (data, protocol, baselines)

Team: _fill in_   |   Repo: _fill in URL_

## 1. Get the data
Download **MovieLens 1M** from https://grouplens.org/datasets/movielens/1m/ (follow GroupLens terms),
unzip it. You need the folder containing `ratings.dat`, `movies.dat`, `users.dat`.

## 2. Install
Python 3.13 (what the pins in `requirements.txt` were tested with).
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Reproduce everything (one command)
```
python run_all.py --data-dir /path/to/ml-1m
```
This uses the default per-user chronological split (`--split-mode per_user`); `--split-mode global` gives the single-cut-date variant. Writes to `outputs/` (change with `--out-dir`):

| file | content |
|---|---|
| `figures/fig_long_tail.png`, `fig_rating_dist.png`, `fig_activity_time.png` | data characterization |
| `tuning_log.csv` | every validation run (full grid, both baselines) |
| `results.csv` | final test results in the layout of the course workbook tab *Part 1 - Baselines* (CIs in the Notes column) |
| `results_long.csv` | same results in long format with CI columns |
| `filter_sensitivity.csv` | effect of item-count filtering on the baselines |
| `summary.json` | all statistics, split details, best configs |
| `report_numbers.md` | every number + draft justification text for the report |

Optional: `python tests/check_metrics.py` verifies the own metric implementation against scikit-learn.

## 4. Protocol (short)
- Per-user chronological split: each user's earliest 80% of ratings train, next 10% validation, last 10% test (`--train-frac`, `--val-frac`). `--split-mode global` uses one cut date instead (leaves most users cold on ML-1M because users rate in bursts).
- Relevant/positive = rating >= 4 (`--pos-threshold`). Every rated item counts as "seen" and is removed from the ranking.
- Ranking over the full catalogue (all items in the data) minus the user's history. P@10, R@10, NDCG@10, Coverage@10.
- Tuning on validation only (NDCG@10); final models refit on train+val and evaluated once on test.
- Evaluated users: have history before the evaluation period and >= 1 positive target item; cold users are counted and excluded.
- Seed: **42** (`--seed`). Everything except the bootstrap CI is deterministic.

## 5. Before submitting
Fill in team name and repo URL above, pin `requirements.txt` from `pip freeze`, and copy the rows of `outputs/results.csv` into the *Part 1 - Baselines* tab of your results workbook.
