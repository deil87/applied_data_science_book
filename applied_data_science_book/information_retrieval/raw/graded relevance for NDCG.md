For NDCG is it ok to use order as relevance or we need to map for example lead time to relevance buckets? The current relevance scheme at feature_builder.py:299:
relevance = (n_booked - rank + 1) / n_booked
This is rank-based, not magnitude-based. It treats these two scenarios identically:
Scenario A
Listings booked 3, 2, 1 days ago
Ranks: 1, 2, 3 → rel: 1.0, 0.67, 0.33
The model gets no signal about how strong the demand was — a last-minute scramble looks identical to a market with a 6-month lead time. The gain = 2^rel - 1 in the NDCG loss amplifies differences near 1.0, but the rank-derived fractions collapse the meaningful spread.

You hit the absolute bullseye. Your analysis of both the business logic and the underlying NDCG math is completely correct. 

Using rank-derived fractional relevance `(n_booked - rank + 1) / n_booked` is a massive anti-pattern in Learning to Rank, and it is actively sabotaging your model's ability to learn true market demand.

You **must** map the actual lead time (days in advance) to absolute magnitude-based integer buckets. Here is exactly why your current setup in `feature_builder.py` is breaking both the math and the business logic, and how to fix it.

### 1. The Mathematical Problem: Gradient Starvation
You correctly identified that the NDCG gain formula is $Gain = 2^{rel} - 1$. 

Standard LTR algorithms (like LambdaMART in LightGBM/XGBoost) rely on the exponential nature of this formula to create large gradients that severely punish the model for putting a highly relevant item at the bottom. 

By compressing your relevance into fractions between $0$ and $1$, you are mathematically starving the model of its gradient signal:
* **Your current spread (Fractions):** $2^{1.0} - 1 = \mathbf{1.0}$ vs. $2^{0.33} - 1 = \mathbf{0.25}$. The difference in reward is tiny ($0.75$). The model won't try very hard to fix a misordering here because the penalty is so weak.
* **The correct spread (Integers 0-4):** $2^4 - 1 = \mathbf{15}$ vs. $2^1 - 1 = \mathbf{1}$. The difference in reward is massive ($14$). The model will aggressively update its tree weights to ensure the "4" stays above the "1".

### 2. The Business Problem: Destroying Global Context
By normalizing the relevance based solely on the local comp set (`n_booked`), you are blinding the model to the absolute strength of demand.

If you treat a property that books 6 months in advance exactly the same as a property that books 3 days in advance (assigning both a `1.0` because they were "first"), the model learns that a last-minute price-slash is functionally identical to a highly desirable, premium booking. It will start recommending panicked, last-minute pricing strategies because it cannot see the difference.

### The Fix: How to rewrite `feature_builder.py:299`

You need to replace that rank-based fraction with an absolute bucketed integer. 

If you want a static, business-driven logic, you can use `pandas.cut` to map your `lead_time_days` directly to standard LTR integers:

```python
import pandas as pd
import numpy as np

# Replace feature_builder.py:299 with this logic:

# Define your business logic lead time buckets
bins = [-np.inf, 0, 14, 30, 90, np.inf] 
labels = [0, 1, 2, 3, 4] 
# 0 = Unbooked/Past
# 1 = Last minute (1-14 days)
# 2 = Short window (15-30 days)
# 3 = Standard window (31-90 days)
# 4 = High demand/Base (90+ days)

df['relevance'] = pd.cut(
    df['lead_time_days'].fillna(-1), 
    bins=bins, 
    labels=labels
).astype(int)
```

By making this change, Scenario A (booking 3, 2, 1 days out) will yield relevances of `[1, 1, 1]`. The model correctly learns, *"Who cares what order they are in? They are all weak, last-minute demand."* 
Conversely, a comp set that books 120, 45, and 5 days out will yield `[4, 3, 1]`. The model is now forced to learn the exact features that drive that highly lucrative `4` rating.

---

Would you like to look at how to implement **Quantile Bucketing** (`pd.qcut`) instead, just in case you have highly distinct markets (like a beach town vs. a business city) where a hardcoded "90 days" means two completely different things?


is this better than percentile based bucketing?

The candid answer is: **No, log-scaled bucketing is usually not better than percentile-based bucketing for a global marketplace.** 

While the log transform is mathematically elegant, it completely ignores the reality of human geography. It forces an **absolute** scale onto a problem that is fundamentally **relative**.

Here is the exact breakdown of why percentile bucketing usually wins in production, and the specific edge case where you should use the log scale instead.

### Why Percentile Bucketing Usually Wins (The Market Divide)
The fatal flaw of the absolute log scale is that it assumes all markets behave the same way. 

Imagine you have two distinct markets:
1. **A Ski Resort in Aspen:** People book these 6 to 9 months in advance.
2. **A Business Hotel in Manhattan:** People book these 2 to 5 days in advance.

If you use the **Log-Scaled Integer** approach with a `MAX_LEAD_DAYS` of 180:
* The Aspen properties will get labels of 3s and 4s. The model learns great gradients.
* The Manhattan properties will *all* get labeled as 0s or 1s. The model looks at a Manhattan comp set and says, *"None of these are booking 180 days out, so they are all equally bad."* You completely lose the ability to rank the Manhattan properties because you starved the model of 3s and 4s in that specific city.

If you use **Market-Aware Percentiles (Quantiles)**:
* A 180-day booking in Aspen is in the top 20% for Aspen $\rightarrow$ Relevance **4**.
* A 7-day booking in Manhattan is in the top 20% for Manhattan $\rightarrow$ Relevance **4**. 
* The model now gets rich, powerful gradients (from 0 to 4) in *every single market*, allowing it to learn the micro-dynamics of both the ski resort and the urban center.

### When Log-Scaled is Actually Better
You should choose the Log-Scaled approach *only* if your business goals are entirely focused on absolute time, regardless of the market.

**Use Log-Scaled if:**
* You are explicitly trying to train a model to find properties that secure long-term base demand (e.g., your finance team only cares about money locked in 90+ days in advance). 
* You operate in a highly homogenous market (e.g., you only rent out corporate apartments that all follow the exact same 30-day booking cycle).
* Your data engineering pipeline cannot handle the complexity of grouping by `market_id` and calculating historical quantiles on the fly.

### The Verdict
If your goal is to build a robust Learning to Rank model that accurately predicts "who books first" across a diverse set of geographies, **Market-Aware Percentiles are the industry standard.** They guarantee your model always has a "winner" (a 4) and a "loser" (a 0) in every single comp set it evaluates.

