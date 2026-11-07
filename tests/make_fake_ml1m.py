"""Writes a small synthetic dataset in MovieLens-1M file format (for smoke tests only)."""
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
n_u, n_i, n_clusters = 900, 1000, 6
pop = 1.0 / np.arange(1, n_i + 1) ** 0.9
rng.shuffle(pop)
cluster_of_item = rng.integers(0, n_clusters, n_i)
genres = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance"]
t_lo = 954547200  # 2000-04-01
rows = []
for u in range(1, n_u + 1):
    c = rng.integers(0, n_clusters)
    w = pop * np.where(cluster_of_item == c, 4.0, 1.0)
    n = int(np.clip(rng.lognormal(4.0, 0.9), 20, 400))
    items = rng.choice(n_i, size=min(n, n_i), replace=False, p=w / w.sum())
    start = t_lo + (rng.integers(0, 270 * 86400) if rng.random() < 0.85 else rng.integers(270 * 86400, 1000 * 86400))
    spread = 3600 if rng.random() < 0.7 else 20 * 86400
    ts = start + np.sort(rng.integers(0, spread, len(items)))
    liked = cluster_of_item[items] == c
    r = np.clip(np.round(rng.normal(3.2 + 0.8 * liked, 1.0)), 1, 5).astype(int)
    for i, rr, t in zip(items, r, ts):
        rows.append((u, i + 1, rr, t))
with open(out / "ratings.dat", "w") as f:
    for u, i, r, t in rows:
        f.write(f"{u}::{i}::{r}::{t}\n")
with open(out / "movies.dat", "w", encoding="latin-1") as f:
    for i in range(n_i):
        f.write(f"{i+1}::Movie {i+1} ({1930 + (i * 7) % 70})::{genres[cluster_of_item[i]]}\n")
with open(out / "users.dat", "w") as f:
    for u in range(1, n_u + 1):
        f.write(f"{u}::{'MF'[u % 2]}::{[1, 18, 25, 35, 45, 50, 56][u % 7]}::{u % 20}::{10000 + u}\n")
print("wrote", out, len(rows), "ratings")
