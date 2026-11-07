# Recommender Systems on MovieLens 1M - Part 1: Data, Protocol and Baselines

Course project, Recommender Systems. This repository contains a fully reproducible evaluation pipeline for MovieLens 1M: data characterization, a leakage-aware temporal split, a full-catalogue ranking evaluation, and two tuned baselines (item-based kNN and Most Popular).

|  |  |
|---|---|
| **Team** | `Group 26` |
| **Members** | `SIVA SANDEEP ATMAKURU` |
| **Status** | Part 1 (data, protocol, baselines) |
| **Seed** | 42 |

## Headline result

Test set, 5,821 evaluated users, ranked against the full 3,706-item catalogue. Brackets are 95% bootstrap intervals over users.

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage@10 |
|---|---|---|---|---|
| Most Popular (all-time, positive only) | 0.0455 | 0.0493 | 0.0584 [0.0553, 0.0616] | 0.039 |
| **Item-kNN (cosine, k = 200)** | **0.0631** | **0.0850** | **0.0883 [0.0842, 0.0921]** | **0.259** |

Item-kNN beats the best popularity variant on every accuracy metric with non-overlapping NDCG intervals, and recommends about seven times as much of the catalogue. All four Most Popular variants and the full tuning grid are in the generated outputs (see below).

## Quick start

```bash
# 1. Get the data (GroupLens terms apply): https://grouplens.org/datasets/movielens/1m/
#    Unzip it. You need the folder that directly contains ratings.dat, movies.dat, users.dat.

# 2. Create an environment and install the pinned dependencies
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Reproduce every number and figure
python run_all.py --data-dir /path/to/ml-1m
```

A full run takes about a minute on a laptop (about 70 s measured) and needs well under 2 GB of RAM. The data path is the only required argument; nothing depends on absolute paths or on files outside this repository.

**Environment.** Developed and tested with Python 3.14 on Windows and Python 3.13 on Linux. Exact package versions are pinned in `requirements.txt`.

## What the pipeline does

1. **Loads and characterizes the data**: users, items, density, long-tail statistics, rating distribution, time span and burstiness (figures and statistics).
2. **Splits chronologically** (80% train / 10% validation / 10% test).
3. **Tunes on validation only**: 18 item-kNN configurations (3 similarities x 6 neighbourhood sizes) and 12 Most Popular configurations (window length x feedback treatment).
4. **Evaluates once on test** after refitting the selected configurations on train + validation, with bootstrap confidence intervals.
5. **Checks sensitivity to filtering**: re-runs the selected models with minimum item-rating thresholds of 5, 10 and 20.
6. **Writes all results** and a file of report-ready numbers.

## Evaluation protocol

| Decision | Choice |
|---|---|
| Split | Per-user chronological: each user's earliest 80% of ratings train, next 10% validation, last 10% test. `--split-mode global` uses a single cut date instead. |
| Why per-user | Most users enter all their ratings in one short session (62.9% within 24 h). A global cut date left only 492 of 6,040 users evaluable on validation because the rest have no earlier history. The per-user split evaluates 5,789 validation and 5,821 test users. Trade-off: no global cut date, so one user's test interaction can precede another user's training interaction. |
| Feedback | Rating >= 4 is positive. Positives define relevant items and the training signal for kNN and positive-only popularity. Every rated item is excluded from that user's ranking. |
| Candidate set | All 3,706 items minus the user's history. No sampled negatives. |
| Metrics | Precision@10, Recall@10, NDCG@10 (binary gains, ideal DCG over min(#relevant, 10)), catalogue coverage@10. Implemented in `metrics.py` and verified against scikit-learn (`tests/check_metrics.py`). |
| Evaluated users | Users with at least one positive item in the target period. Users with no earlier history would be counted and excluded (none occur with the per-user split). |
| Tie-breaking | Deterministic, by training popularity. |
| Intervals | 95% percentile bootstrap over users, 1,000 resamples, seed 42. |
| Tuning | Validation only (NDCG@10). Final models are refit on train + validation and scored once on test. |

## Baselines

- **Item-based kNN.** Binary positive user-item matrix. The score of item *j* for user *u* is the sum of the similarities between *j* and the items in *u*'s positive history, keeping only each target item's *k* most similar items (negative similarities clipped to 0). Similarities: cosine, Jaccard, adjusted cosine; k in {10, 20, 50, 100, 200, 500}.
- **Most Popular.** Four variants: all-time or recent window (7, 30, 90, 180, 365 days before the end of the fitted data) crossed with counting all ratings or only positive ones.

## Repository layout

```
run_all.py          single entry point; runs everything
data.py             loading, filtering, temporal splits, interaction matrices
eda.py              data statistics and figures
models.py           MostPopular and ItemKNN
metrics.py          ranking metrics, full-catalogue evaluation, bootstrap CIs
results.py          results writers (course workbook layout)
report.py           generates outputs/report_numbers.md
requirements.txt    pinned dependencies
tests/
  check_metrics.py  verifies metrics against scikit-learn and direct definitions
  make_fake_ml1m.py synthetic data in ML-1M format, for smoke tests
```

## Outputs (written to `outputs/`)

| File | Content |
|---|---|
| `results.csv` | Final test results in the layout of the course workbook tab *Part 1 - Baselines*; confidence intervals are in the Notes column |
| `results_long.csv` | The same results in long format with explicit CI columns |
| `tuning_log.csv` | Every validation run of the full tuning grid |
| `filter_sensitivity.csv` | Effect of item-count filtering on the selected models |
| `summary.json` | All statistics, split details, selected configurations |
| `report_numbers.md` | Every number used in the report |
| `figures/` | `fig_long_tail.png`, `fig_rating_dist.png`, `fig_activity_time.png` |

## Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--data-dir` | required | Folder with `ratings.dat`, `movies.dat`, `users.dat` |
| `--out-dir` | `outputs` | Where results are written |
| `--split-mode` | `per_user` | `per_user` or `global` |
| `--train-frac`, `--val-frac` | 0.8, 0.1 | Split proportions |
| `--pos-threshold` | 4 | Ratings at or above this are positive |
| `--min-user`, `--min-item` | 0, 0 | Minimum ratings per user / item for the main run |
| `--seed` | 42 | Random seed (used for bootstrap resampling) |
| `--n-boot` | 1000 | Bootstrap resamples |
| `--quick` | off | Tiny grids, for smoke tests only |

## Reproducibility

- One command regenerates every reported number and figure; the data path is configurable.
- Dependencies are pinned in `requirements.txt`.
- Both baselines are deterministic. The only randomness is the bootstrap, seeded with 42.
- No intermediate files are required: everything is recreated from the raw data.
- Sanity check without the real data: `python tests/make_fake_ml1m.py fake_ml1m && python run_all.py --data-dir fake_ml1m --out-dir out_fake`.

## Known limitations

- The per-user split respects each user's own time order but not a global one, which may make results somewhat optimistic.
- Because most users rate in a single session, test performance mostly reflects within-session behaviour rather than long-term preference drift.
- Part 1 uses a single data split; intervals reflect user sampling only. Paired tests and repeated runs are planned for Part 2.
- The explanation for recent-window popularity underperforming all-time popularity is a hypothesis and has not been tested.

## Data and acknowledgements

MovieLens 1M is provided by GroupLens Research and is not redistributed here; download it from the link above and follow its usage terms. Harper, F. M. and Konstan, J. A. (2015), *The MovieLens Datasets: History and Context*, ACM TiiS 5(4).

## AI assistance

AI tools were used in this project. Claude (Anthropic) assisted with writing the code and drafting text; see the AI-use section of the project report for details. <!-- Edit to match your report. -->
