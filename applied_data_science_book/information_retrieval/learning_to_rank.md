---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Learning to Rank

**Learning to Rank (LTR)** is the application of supervised machine learning to
train a ranking model from labelled query–document pairs. It bridges classical IR
(where relevance functions are hand-crafted) and modern neural approaches.

## Problem framing

Given a query `q` and a candidate set of documents `{d_1, …, d_n}`, learn a
scoring function `f(q, d)` such that sorting by `f` maximises a ranking metric
(NDCG, MAP, MRR) on held-out queries.

### Three paradigms

| Paradigm | Loss | Typical model | Benchmark dataset |
|---|---|---|---|
| **Pointwise** | Regression / classification on relevance label | Gradient boosted trees | — |
| **Pairwise** | Prefer `d_i > d_j` when `rel(d_i) > rel(d_j)` | RankSVM, RankNet | LETOR |
| **Listwise** | Directly optimise a list-level metric | LambdaMART, ListNet | MSLR-WEB10K |

Listwise methods — especially **LambdaMART** — dominate classic LTR benchmarks
and are still the industry default for production ranking systems.

## Relevance labels: getting them right

The quality of your relevance labels is the single most important factor in
training a good LTR model. A surprisingly common mistake is using
**rank-derived fractional relevance** instead of **magnitude-based integer
buckets**, and it silently sabotages model training.

### The anti-pattern: rank-based fractions

Consider a scheme like:

```python
relevance = (n_booked - rank + 1) / n_booked
```

This maps the first-booked item to `1.0`, the second to `0.67`, the third to
`0.33`, and so on — regardless of *when* they were booked. Two very different
scenarios become indistinguishable:

| Scenario | Bookings | Rank-derived relevances |
|---|---|---|
| A — last-minute market | 3, 2, 1 days before arrival | 1.0, 0.67, 0.33 |
| B — high-demand market | 120, 45, 5 days before arrival | 1.0, 0.67, 0.33 |

The model gets no signal about absolute demand strength. A panicked last-minute
price-slash looks functionally identical to a premium booking secured months
in advance.

### Why fractions break NDCG gradients

LambdaMART and other listwise algorithms derive their update signal from the
NDCG gain formula $\text{Gain} = 2^{\text{rel}} - 1$. The exponential is
intentional: it creates **large gradients** that force the model to aggressively
correct misordered highly-relevant items.

With fractional labels in $[0, 1]$ the gain spread collapses:

| Relevance scheme | Best gain | Worst gain | Spread |
|---|---|---|---|
| Fractions (0–1) | $2^{1.0}-1 = 1.0$ | $2^{0.33}-1 = 0.25$ | **0.75** |
| Integer buckets (0–4) | $2^4 - 1 = 15$ | $2^1 - 1 = 1$ | **14** |

The model trained on fractions sees a penalty 18× smaller for a mislabelled
result. It will not try hard to fix misordered items — this is **gradient
starvation**.

### The fix: absolute integer buckets

Replace the rank-based fraction with a business-driven bucketing of the
underlying continuous signal (e.g. booking lead time in days):

```python
import pandas as pd
import numpy as np

bins   = [-np.inf, 0, 14, 30, 90, np.inf]
labels = [0, 1, 2, 3, 4]
# 0 = Unbooked / past
# 1 = Last minute    (1–14 days)
# 2 = Short window   (15–30 days)
# 3 = Standard window(31–90 days)
# 4 = High demand    (90+ days)

df['relevance'] = pd.cut(
    df['lead_time_days'].fillna(-1),
    bins=bins,
    labels=labels,
).astype(int)
```

With this scheme Scenario A yields `[1, 1, 1]` — the model correctly learns
*"all of these are weak last-minute bookings"* — while Scenario B yields
`[4, 3, 1]`, giving the model rich gradient signal to learn what drives premium
demand.

### Static buckets vs. percentile (market-aware) bucketing

Static thresholds (like the 90-day cutoff above) assume all markets behave
similarly. In a heterogeneous marketplace this breaks down:

| Market | Typical booking window | Static bucket for a 7-day booking |
|---|---|---|
| Ski resort in Aspen | 6–9 months ahead | 1 (last-minute) |
| Business hotel in Manhattan | 2–5 days ahead | 1 (last-minute) |

With static buckets both markets assign `1` to their best bookings. The
Manhattan comp-set never sees a `3` or `4` — gradient starvation returns for
that entire city.

**Percentile (quantile) bucketing** solves this by computing bucket boundaries
relative to each market's historical distribution:

```python
df['relevance'] = (
    df.groupby('market_id')['lead_time_days']
      .transform(lambda x: pd.qcut(x, q=5, labels=[0, 1, 2, 3, 4],
                                   duplicates='drop'))
      .astype(int)
)
```

Now a 7-day booking in Manhattan that is in the top 20% for that market
receives a `4` — the same gradient weight as a 180-day Aspen booking. Every
comp-set always has a winner and a loser, maximising the learning signal across
all markets.

**When to prefer static buckets:** your business objective is explicitly
absolute (e.g. "only care about bookings locked in 90+ days in advance"), your
market is highly homogeneous, or computing per-market quantiles is
operationally infeasible.

**When to prefer percentile bucketing:** you operate across diverse geographies
or categories and want the model to learn relative demand dynamics within each
context.

## LambdaMART

LambdaMART combines:
- **Lambda gradients** — pseudo-gradients derived from pairwise preference
  swaps, weighted by the change in NDCG they would produce.
- **MART** (Multiple Additive Regression Trees) — gradient boosted decision
  trees (GBDT).

Implementations: `lightgbm` (`objective='lambdarank'`),
`xgboost` (`objective='rank:ndcg'`), `ranklib`.

```python
import lightgbm as lgb

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 5, 10],
    "num_leaves": 64,
    "learning_rate": 0.05,
}

train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
model = lgb.train(params, train_data, num_boost_round=200)
```

## Neural rankers

**Cross-encoders** (e.g. fine-tuned BERT) jointly encode query + document and
output a relevance score. They are more accurate than bi-encoders but far slower
(no pre-indexing), so they are typically used as a **re-ranker** on the top-k
candidates returned by a first-stage retriever (BM25 or dense).

```
query + candidate → [CLS] token → linear head → relevance score
```

## Two-stage pipeline

```
Stage 1 — Retrieval (fast, high recall)
  BM25 or dense retriever → top-1000 candidates

Stage 2 — Re-ranking (slow, high precision)
  LambdaMART or cross-encoder → final top-10
```

## Custom loss functions

The existing notebook (`Learning to rank/LTR with custom loss function.ipynb`)
demonstrates how to implement a bespoke ranking loss in a GBDT framework,
allowing direct optimisation of business-specific objectives beyond standard
NDCG.

Key ideas:
- Replace lambda gradients with domain-specific pair weights.
- Incorporate business constraints (diversity, freshness, inventory) into the
  loss.
- Evaluate offline (NDCG) and online (CTR, conversion) in tandem.

## Public datasets

| Dataset | Queries | Docs per query | Relevance |
|---|---|---|---|
| MSLR-WEB10K | 10,000 | ~120 | 0–4 graded |
| MSLR-WEB30K | 30,000 | ~120 | 0–4 graded |
| Yahoo! LTR (C14) | 29,921 | ~24 | 0–4 graded |
| ISTELLA | 33,018 | ~103 | 0–4 graded |

The `MSLR-WEB10K/` dataset is already present in
`applied_data_science_book/Learning to rank/MSLR-WEB10K/`.
