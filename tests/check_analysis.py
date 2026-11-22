"""Checks the Part 2 analysis helpers against brute-force loops.

    python tests/check_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analysis  # noqa: E402

rng = np.random.default_rng(1)
U, I = 60, 40
rel = rng.random((U, I)) < 0.1
scores = rng.random((U, I))
topk = np.argsort(-scores, axis=1)[:, :10]
mask = rng.random(I) < 0.3
rec, ok, share = analysis.item_group_metrics(topk, rel, mask)
tot_slots = in_group_slots = 0
for u in range(U):
    rel_items = [j for j in range(I) if rel[u, j] and mask[j]]
    assert ok[u] == (len(rel_items) > 0)
    if rel_items:
        hits = len([j for j in topk[u] if rel[u, j] and mask[j]])
        assert abs(rec[u] - hits / len(rel_items)) < 1e-12
    in_group_slots += sum(mask[j] for j in topk[u])
    tot_slots += 10
assert abs(share - in_group_slots / tot_slots) < 1e-12
print("item_group_metrics matches the brute-force loop. OK")

a = rng.normal(0.5, 1, 4000)
b = a - 0.2 + rng.normal(0, 0.1, 4000)  # b is lower by about 0.2 and paired with a
d, lo, hi = analysis.paired_diff_ci(a, b, 500, 0)
assert lo < 0.2 < hi and lo > 0, (d, lo, hi)
d0, lo0, hi0 = analysis.paired_diff_ci(a, a + rng.normal(0, 0.01, 4000), 500, 0)
assert lo0 < 0 < hi0
print("paired_diff_ci: detects a real shift (%.3f [%.3f, %.3f]) and no shift when there is none. OK" % (d, lo, hi))

df = pd.DataFrame({"u": rng.integers(0, 5, 1000), "i": rng.integers(0, 7, 1000), "r": 5, "ts": 0})
assert len(analysis.subsample(df, 0.2, 3)) == 200 and len(analysis.subsample(df, 1.0, 3)) == 1000
assert analysis.subsample(df, 0.5, 3).equals(analysis.subsample(df, 0.5, 3))
print("subsample keeps the requested fraction and is reproducible. OK")

low, high, med = analysis.activity_groups(df, np.arange(5), 5)
assert (low ^ high).all()
head, head_min = analysis.item_groups(df, 7, 0.2)
assert head.sum() == 1
print("activity and item groups are complete and disjoint. OK")
