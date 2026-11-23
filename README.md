# Recommender Systems: Models and Honest Comparison on MovieLens 1M

> Course project, Group 26 · Part 2 (Models and Honest Comparison) with Part 1 retained · Part 3 track: **A – Diversity and Popularity Bias**

A controlled comparison of six recommenders on MovieLens 1M under a single evaluation protocol, with equal tuning effort, multiple seeds, paired confidence intervals, group-level breakdowns and a data-sparsity test. Two of the models, **BPR-MF** and **EASE**, are implemented from scratch in NumPy.

---

## Table of Contents

1. [Headline Results](#headline-results)
2. [Models](#models)
3. [Experimental Protocol](#experimental-protocol)
4. [Key Findings](#key-findings)
5. [Repository Structure](#repository-structure)
6. [Getting Started](#getting-started)
7. [Reproducing the Results](#reproducing-the-results)
8. [Limitations](#limitations)
9. [Roadmap: Part 3](#roadmap-part-3)
10. [Use of AI Assistants](#use-of-ai-assistants)
11. [References](#references)
12. [Author](#author)

---

## Headline Results

Test results on 5,821 users, full-catalogue ranking (3,706 items), cutoff 10. NDCG@10 shown with 95% bootstrap interval over users.

| Model | P@10 | R@10 | NDCG@10 [95% CI] | Coverage@10 |
|---|---|---|---|---|
| Most Popular (all-time, positive only) | 0.0455 | 0.0493 | 0.0584 [0.0553, 0.0616] | 0.039 |
| Item-kNN (cosine, k = 200) | 0.0631 | 0.0850 | 0.0883 [0.0842, 0.0921] | 0.259 |
| User-kNN (Jaccard, k = 50, shrink 30) | 0.0642 | 0.0887 | 0.0910 [0.0867, 0.0948] | 0.187 |
| ALS (64 factors, reg 1, alpha 5) | 0.0594 | 0.0922 | 0.0882 [0.0845, 0.0919] | 0.316 |
| BPR-MF (own) | 0.0585 | 0.0817 | 0.0822 [0.0790, 0.0852] | 0.353 |
| **EASE (own, λ = 750)** | **0.0709** | **0.1032** | **0.1014 [0.0971, 0.1054]** | 0.226 |

**EASE is the most accurate model.** Its paired NDCG@10 advantage over the runner-up (user-kNN) is **+0.0104 [+0.0079, +0.0130]**, and the interval excludes zero.

---

## Models

| Model | Paradigm | Implementation | Own code |
|---|---|---|---|
| Most Popular | Non-personalised | Part 1 (12 configurations: window × feedback type) | – |
| Item-kNN | Neighbourhood CF | Part 1 (cosine / Jaccard / adjusted cosine, k ∈ {10 … 500}) | – |
| User-kNN | Neighbourhood CF | NumPy/SciPy, no library | – |
| ALS | Latent-factor CF | [`implicit`](https://github.com/benfred/implicit) library | – |
| **BPR-MF** | Latent-factor, ranking loss | `bpr.py`, NumPy only | ✅ |
| **EASE** | Linear item-item autoencoder | `ease.py`, NumPy only | ✅ |

**BPR-MF.** Score `x_ui = p_u · q_i + b_i`, trained by mini-batch SGD on sampled triples (u, i, j) with hand-derived gradients. Verified against finite differences (max error 3e-10).

**EASE.** Closed-form ridge regression with a zero-diagonal constraint: `B[i,j] = -P[i,j] / P[j,j]` where `P = (XᵀX + λI)⁻¹`. Deterministic, one hyperparameter. Verified against explicit per-item constrained regressions (agreement to 1e-8).

---

## Experimental Protocol

- **Data:** MovieLens 1M, 1,000,209 ratings, 6,040 users, 3,706 items. Ratings ≥ 4 are positive feedback.
- **Split:** per-user chronological 80 / 10 / 10 (train / validation / test). Models are tuned on validation, refit on train + validation, and evaluated **once** on test.
- **Ranking:** full catalogue, user history excluded, **no sampled negatives**.
- **Metrics:** Precision@10, Recall@10, NDCG@10, Coverage@10. Implemented in-house and verified against `scikit-learn`'s `ndcg_score`.
- **Uncertainty:** 95% percentile bootstrap over users (1,000 resamples). Model differences use **paired** bootstraps on the same users. Seeds 42–46 for stochastic models.
- **Tuning budget:** 18 configurations per family (or the full grid if smaller), validation only.

> A global temporal split was tried first and rejected: 90.5% of ratings were entered in 2000 and most users rated in a single sitting, leaving only 492 evaluable validation users.

---

## Key Findings

1. **EASE leads on accuracy**, and keeps its lead at every training fraction (100%, 50%, 20%).
2. **User-kNN, item-kNN and ALS are statistically tied.** All paired intervals include zero.
3. **Our BPR trails the kNN models and ALS.** The best setting sits at the edge of the grid, so it is likely under-tuned.
4. **The accuracy winner is not the long-tail winner.** EASE spends 98.7% of its top-10 slots on head items; BPR (tail R@10 = 0.0100) and ALS (0.0093) find more of the tail than EASE (0.0056).
5. **Method rankings depend on user activity.** ALS is second-best for low-activity users but fifth for high-activity users.
6. **Sparsity breaks most models.** With 20% of the training data, item-kNN, user-kNN, ALS and BPR all fall below plain popularity. Only EASE stays ahead.

| NDCG@10 (rank of 6) | 100% | 50% | 20% |
|---|---|---|---|
| Most Popular | 0.0584 (6) | 0.0589 (6) | 0.0582 (2) |
| Item-kNN | 0.0883 (3) | 0.0803 (2) | 0.0453 (6) |
| User-kNN | 0.0910 (2) | 0.0754 (3) | 0.0517 (4) |
| ALS | 0.0882 (4) | 0.0695 (5) | 0.0472 (5) |
| BPR (own) | 0.0822 (5) | 0.0698 (4) | 0.0535 (3) |
| EASE (own) | **0.1014 (1)** | **0.0857 (1)** | **0.0642 (1)** |

---

## Repository Structure

> Adjust folder and file names to match your repository.

```text
Recommender_System-Group_26/
├── README.md
├── requirements.txt
├── data/                  # MovieLens 1M (not committed, see Getting Started)
├── src/
│   ├── data.py            # loading, preprocessing, per-user chronological split
│   ├── metrics.py         # P@10, R@10, NDCG@10, coverage, bootstrap
│   ├── baselines.py       # Most Popular, item-kNN
│   ├── userknn.py         # user-based kNN
│   ├── als_model.py       # wrapper around implicit ALS
│   ├── bpr.py             # BPR-MF (own implementation)
│   └── ease.py            # EASE (own implementation)
├── tests/                 # gradient checks, metric checks, EASE closed-form check
├── results/
│   └── tuning_log.csv     # full tuning grid for every model
├── figures/
└── report/
    └── Report_Part2.pdf
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Packages: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `implicit`

### Installation

```bash
git clone https://github.com/Siva-Sandeep7/Recommender_System-Group_26.git
cd Recommender_System-Group_26

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data

Download **MovieLens 1M** from [GroupLens](https://grouplens.org/datasets/movielens/1m/) and extract `ratings.dat` into `data/`.

---

## Reproducing the Results

> Replace the commands below with your actual entry points.

```bash
# 1. Run unit tests (gradient checks, metric checks, EASE closed form)
pytest tests/

# 2. Tune all models on validation
python src/run_tuning.py --seed 42

# 3. Refit on train + validation and evaluate on test (seeds 42-46)
python src/run_final.py

# 4. Disaggregated analysis and sparsity ablation
python src/run_analysis.py
python src/run_sparsity.py
```

The full tuning grid is recorded in `results/tuning_log.csv`.

---

## Limitations

- **Single split.** Intervals cover user sampling and seed variation, not split or tuning variation.
- **Per-user split can be optimistic**, and because most users rate in one session, test performance mainly measures within-session prediction.
- **ALS and BPR** were searched over random subsets of their grids; BPR is probably under-tuned.
- **EASE was added after** the first full run showed BPR trailing; its λ was chosen on validation only.
- **Hyperparameters were not retuned** for the sparsity runs, which may understate every model.
- **No metadata** (genres, user attributes) is used by any model.

---

## Roadmap: Part 3

**Track A: Diversity and popularity bias**, with EASE as the base recommender.

- [ ] MMR re-ranking with genre vectors as item representation
- [ ] Sweep of the trade-off parameter to draw the accuracy-diversity frontier
- [ ] Evaluation of intra-list diversity, coverage, Gini index and calibration error

---

## Use of AI Assistants

Claude (Anthropic) was used to write and debug code, write tests, analyse result files and draft the report. The author ran all code, checked the results and text, and is responsible for everything submitted.

---

## References

- F. M. Harper and J. A. Konstan. *The MovieLens Datasets: History and Context.* ACM TiiS, 5(4), 2015.
- B. Sarwar, G. Karypis, J. Konstan and J. Riedl. *Item-based collaborative filtering recommendation algorithms.* WWW, 2001.
- S. Rendle, C. Freudenthaler, Z. Gantner and L. Schmidt-Thieme. *BPR: Bayesian personalized ranking from implicit feedback.* UAI, 2009.
- Y. Hu, Y. Koren and C. Volinsky. *Collaborative filtering for implicit feedback datasets.* ICDM, 2008.
- H. Steck. *Embarrassingly shallow autoencoders for sparse data.* WWW, 2019.

---

## Author

**Siva Sandeep Atmakuru** · Group 26
Repository: <https://github.com/Siva-Sandeep7/Recommender_System-Group_26>
