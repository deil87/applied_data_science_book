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

# Causal Inference

> *"Correlation is not causation."* — Every statistician, always.

Standard machine learning excels at prediction: given features $X$, estimate $E[Y|X]$ as accurately as possible. But many of the most important business questions are *causal*: not "which customers are likely to churn?" but "for which customers does sending a retention offer actually *reduce* churn?"

These are fundamentally different questions. A predictive model that tells you a customer has a 90% churn probability gives you no information about whether your intervention will help. Causal inference provides the framework to answer the intervention question directly.

---

## Why Causal Inference in Applied Data Science?

In industry, decisions are actions — discounts, emails, drugs, policy changes. The relevant quantity is always the **effect of the action**, not the baseline prediction. Causal inference methods allow us to:

- Estimate the effect of a treatment or intervention from observational data (without running an expensive randomized trial).
- Identify *which individuals* benefit most from an intervention — not just whether it works on average.
- Avoid **selection bias**: the systematic difference between who gets treated and who doesn't in the real world.

---

## Key Concepts

### Potential Outcomes Framework

For each individual $i$, define two **potential outcomes**:

- $Y_i(1)$: the outcome if individual $i$ is treated ($T=1$).
- $Y_i(0)$: the outcome if individual $i$ is not treated ($T=0$).

The **Individual Treatment Effect (ITE)** is:

$$\tau_i = Y_i(1) - Y_i(0)$$

The fundamental problem of causal inference is that we only ever observe *one* of these — we never see both $Y_i(1)$ and $Y_i(0)$ for the same person at the same time. Causal inference methods are strategies for estimating the unobserved counterfactual.

### Average Treatment Effect (ATE) vs. CATE

| Quantity | Formula | Question answered |
|:---|:---|:---|
| **ATE** | $E[\tau_i] = E[Y(1) - Y(0)]$ | Does the treatment work *on average*? |
| **ATT** | $E[\tau_i \mid T=1]$ | Does it work for those who *were* treated? |
| **CATE** | $\tau(x) = E[Y(1) - Y(0) \mid X=x]$ | Does it work for individuals *like this*? |

The **Conditional Average Treatment Effect (CATE)** is the workhorse of personalized decision-making. Instead of a single number, CATE is a function of individual features $X$ — it tells you the expected effect for a person with a particular profile.

### Confounding and Selection Bias

In a randomized controlled trial (RCT), treatment is assigned randomly, so $T$ is independent of $Y(0)$ and $Y(1)$. In observational data, this is rarely true. Customers who receive a loyalty email may have already been more likely to book. Patients who get a drug may already be sicker. This **confounding** biases naive comparisons.

Causal methods handle confounding through a set of identification assumptions — most commonly:

- **Unconfoundedness (Ignorability):** $(Y(0), Y(1)) \perp T \mid X$ — conditional on observed features, treatment is as-good-as-random.
- **Overlap (Positivity):** $0 < P(T=1 \mid X) < 1$ — every individual has a nonzero chance of being in either group.

---

## The Meta-Learner Family

A **meta-learner** is a strategy that uses standard supervised learning models as building blocks to estimate CATE. Rather than inventing a new algorithm from scratch, meta-learners orchestrate existing models (gradient boosting, random forests, neural networks) to produce causal estimates.

The four main meta-learners differ in *how* they use the treatment variable:

| Learner | Core idea | Key limitation |
|:---|:---|:---|
| **S-Learner** | One model with $T$ as a feature | May ignore $T$ if its signal is weak |
| **T-Learner** | Separate models for treated/control | High variance when one group is small |
| **X-Learner** | Imputes individual effects; iterates | Complex; adds assumptions |
| **DR-Learner** | Combines outcome models + propensity weighting | Robust to misspecification of one nuisance model |

---

## Sections in This Chapter

```{tableofcontents}
```
