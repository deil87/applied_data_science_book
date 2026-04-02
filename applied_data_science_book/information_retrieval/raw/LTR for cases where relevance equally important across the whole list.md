The problem with NDCG@5 for your use case
Your concern is exactly right. Here's the specific mismatch:
What NDCG@5 optimises
NDCG@k is a top-heavy metric. It applies a logarithmic discount 1/log₂(1+rank) so that rank 1 contributes ~3× more signal than rank 4, and items beyond rank k contribute zero signal. The training loss (_approx_ndcg_loss, line 69) and evaluation (compute_ndcg(..., k=5), line 399) are both purely top-5 focused.
What you actually need
You want to predict the absolute rank of your anchor listing — even when it sits at position 8, 12, or 18 in a 20-listing market. NDCG@5 is completely blind to the difference between rank 6 and rank 20. A model trained purely on NDCG@5 loss will correctly order the top 5 booked listings but will have arbitrary ordering for everything below rank 5.
---
Candidate metrics / losses to consider
Here's an honest comparison of the options that are viable within your current GA2M + TF setup:
1. MRR (Mean Reciprocal Rank) — probably not
Cares only about the rank of the first relevant item. Even more top-heavy than NDCG. Worse for your case.
2. Full-list NDCG (no cutoff k) — marginal improvement
Your current _approx_ndcg_loss already computes over the full list (no cutoff — it sums all positions). The k=5 is only in the evaluation compute_ndcg. So training is already full-list. The evaluation metric is misleadingly truncated. Changing k=5 → k=LIST_SIZE in the eval call is a low-cost fix but doesn't change what the model actually learns.
3. Kendall's τ / Spearman's ρ — conceptually right, hard to optimise
Both measure the full-list ordering agreement. But neither is differentiable — you'd need a surrogate like ListMLE or LambdaRank to approximate them. Adds significant complexity.
4. LambdaRank with position-aware weights — best fit
LambdaRank (Burges et al., 2006) extends pairwise learning-to-rank by weighting each pair (i,j) by the NDCG delta that would result from swapping them. Critically, it can be configured to care equally about all positions (by using full-list NDCG delta rather than NDCG@k delta). This is the most principled approach for "I care about the full ordering, not just the top."
5. MSE / Huber regression on predicted rank — simple and surprisingly effective
Train the model to predict the actual rank (1, 2, 3, … N) directly as a regression target, rather than graded relevance labels. Loss = MSE or Huber between predicted score rank and true rank. This is:
- Perfectly position-aware by construction
- Trivially differentiable
- Easy to evaluate with Spearman's ρ or MAE on anchor-listing rank
- No need for the quartile bucketing scheme at all (your daysAgoBooked → p25/p50/p75 → 0-4 label conversion)
The downside: you lose the explicit "booked sooner = more relevant" signal and instead treat all rank positions equally even if the difference between rank 1 and rank 2 is more meaningful than between rank 10 and rank 11.