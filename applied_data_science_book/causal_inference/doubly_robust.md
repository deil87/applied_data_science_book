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

# CATE and the Doubly Robust Learner

## What is CATE?

The **Conditional Average Treatment Effect (CATE)** is the expected causal effect of a treatment for an individual with feature vector $X = x$:

$$\tau(x) = E\bigl[Y(1) - Y(0) \mid X = x\bigr]$$

Unlike the population-level ATE, CATE is a *function* of individual characteristics. This makes it the cornerstone of personalised decision-making:

- In marketing: who gains the most from a promotional email?
- In medicine: which patients respond to a given drug?
- In policy: for which households does a subsidy change behaviour?

Knowing CATE lets you **target** — focus interventions on the individuals where they create real value, and withhold them from those who would respond poorly or not at all.

---

## The Meta-Learner Family: Where Does DR Fit?

Meta-learners are strategies that assemble standard supervised-learning models to estimate CATE. They differ in *how* they use the treatment variable $T$.

```{mermaid}
flowchart TD
    DATA["Observational Data\n(X, T, Y)"]

    DATA --> SL
    DATA --> TL
    DATA --> XL
    DATA --> DR

    subgraph SL["S-Learner (Single model)"]
        direction TB
        S1["One model μ(X, T)\nT is just another feature"]
        S2["CATE = μ(X,1) − μ(X,0)"]
        S1 --> S2
    end

    subgraph TL["T-Learner (Two models)"]
        direction TB
        T1["μ̂₀(X) — trained on control group\nμ̂₁(X) — trained on treated group"]
        T2["CATE = μ̂₁(X) − μ̂₀(X)"]
        T1 --> T2
    end

    subgraph XL["X-Learner (Iterative)"]
        direction TB
        X1["Built on top of T-Learner\nImputes individual effects\nusing cross-group predictions"]
        X2["CATE via weighted blend\nof imputed effects"]
        X1 --> X2
    end

    subgraph DR["DR-Learner (Doubly Robust)"]
        direction TB
        DR1["Outcome models: μ̂₀(X), μ̂₁(X)\n(T-Learner style)"]
        DR2["Propensity model: ê(X)\nP(T=1 | X)"]
        DR3["Pseudo-outcome Y*\ncombines both corrections"]
        DR4["Final model ψ(X)\npredicts CATE from Y*"]
        DR1 --> DR3
        DR2 --> DR3
        DR3 --> DR4
    end

    TL -. "inherits outcome\nmodel structure" .-> XL
    TL -. "inherits outcome\nmodel structure" .-> DR
    DR -. "adds propensity\nweighting on top" .-> DR

    style DR fill:#d4edda,stroke:#28a745,color:#000
    style TL fill:#cce5ff,stroke:#004085,color:#000
    style XL fill:#fff3cd,stroke:#856404,color:#000
    style SL fill:#f8d7da,stroke:#721c24,color:#000
```

**Key observations from the diagram:**

- The **S-Learner** treats $T$ as just another input column — simple but can be fooled into ignoring treatment if its signal is weak.
- The **T-Learner** separates the data into two subsets and trains a model on each — no $T$ column needed, but small group sizes hurt.
- The **X-Learner** extends T-Learner by imputing individual treatment effects via cross-group prediction — better when groups are unbalanced.
- The **DR-Learner** inherits the T-Learner's outcome models but *adds* a propensity model. These two correction streams are combined into a single pseudo-outcome target. DR is the only meta-learner with a formal double-robustness guarantee.

---

## How the DR-Learner Works

The DR-Learner operates as a three-phase pipeline. It does not just predict $Y$ — it predicts the *causal lift* directly, using two internal models to cross-correct each other.

### Inputs

Every row in your dataset provides three things:

| Symbol | Meaning | Example (hotel campaign) |
|:---|:---|:---|
| $X_i$ | Individual features | Age, loyalty tier, previous stays |
| $T_i \in \{0,1\}$ | Treatment indicator | Did the guest receive the flash-sale email? |
| $Y_i$ | Observed outcome | Did the guest book? ($1$ = yes, $0$ = no) |

The composition of $X$ matters: confounders must be present for unbiased estimation, effect modifiers must be present for heterogeneous CATE predictions, and bad controls (mediators, colliders) must be excluded. See {doc}`feature_roles` for the full breakdown.

---

### Phase 1 — Nuisance Models

Before estimating any causal effect, the DR-Learner trains three **nuisance models** — so called because their predictions are intermediary; we do not report them as the final answer.

#### Outcome models (T-Learner style)

Two separate regression models are trained on *segregated* data — treatment is not a column, it is a *split*:

$$\hat{\mu}_0(X) = E[Y \mid X,\ T=0] \qquad \text{(trained on control group only)}$$

$$\hat{\mu}_1(X) = E[Y \mid X,\ T=1] \qquad \text{(trained on treated group only)}$$

$\hat{\mu}_1(X_i)$ answers: *"What would this person's outcome be if they were treated?"*

#### Propensity model

A classifier $\hat{e}(X)$ is trained on the entire dataset to predict the probability that each individual was treated:

$$\hat{e}(X) = P(T=1 \mid X)$$

This is usually a logistic regression or a gradient-boosted classifier. The propensity score captures *selection bias* — the systematic way in which treated and control individuals differ.

```{admonition} Why do we need both?
:class: tip

The outcome models tell us what *should* happen. The propensity model tells us *how surprising* the actual assignment was. Together they enable the doubly robust correction in Phase 2.
```

---

### Phase 2 — The Pseudo-Outcome

This is the core of the DR-Learner. For every observation, we synthesise a new target value $Y^*_i$ — a **pseudo-outcome** that incorporates both corrections:

$$\boxed{Y^*_i = \hat{\mu}_1(X_i) + \frac{T_i\,(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)}}$$

Decompose it term by term:

| Term | Meaning |
|:---|:---|
| $\hat{\mu}_1(X_i)$ | **Base:** the outcome model's prediction for the treated potential outcome. |
| $Y_i - \hat{\mu}_1(X_i)$ | **Residual:** how much the actual outcome surprised the model. |
| $T_i \cdot (\cdots)$ | **Gate:** for control units ($T=0$), the correction term vanishes; only the base is used. |
| $/ \hat{e}(X_i)$ | **Inverse propensity weight:** amplifies the correction for individuals who were unlikely to be treated — rare treated units carry more information. |

**Intuition:** Start with the regression's best guess. If the person was actually treated, check whether the outcome was surprising. If it was, and if that person was unlikely to be treated (rare), amplify the correction. Rare but surprising treated outcomes are strong evidence about the true causal effect.

---

### Phase 3 — The Final CATE Model

The pseudo-outcomes $Y^*_i$ serve as labels for a final supervised model $\psi$:

$$\hat{\tau}(X) = \psi(X) \approx E[Y^* \mid X]$$

- **Input:** $X$ (original individual features)
- **Target:** $Y^*$ (doubly-robust pseudo-outcomes)
- **Output:** $\hat{\tau}(X)$ — the estimated CATE

Any standard regression model can be used for $\psi$: gradient-boosted trees, random forests, linear models, neural networks. The causal structure is baked into $Y^*$, so $\psi$ is just fitting a regression.

---

## The Double Robustness Property

The DR-Learner's most important theoretical property is **double robustness**: the estimator is consistent for the true CATE *even if one of the two nuisance models is misspecified* — as long as the other is correct.

Formally:

$$\hat{\tau}(x) \xrightarrow{p} \tau(x) \quad \text{if either } \hat{\mu}_1 \text{ is consistent OR } \hat{e} \text{ is consistent (or both)}$$

### Why this matters in practice

In real-world observational data, you are almost never certain that any single model is perfectly specified. Confounders may be partially observed, functional forms are guessed, and sample sizes are finite. Double robustness provides a safety net:

| Scenario | What saves you |
|:---|:---|
| Propensity model is badly calibrated | A well-specified outcome model $\hat{\mu}_1$ anchors $Y^*$ to the right value. |
| Outcome model is badly specified | A well-calibrated propensity $\hat{e}$ re-weights the residuals correctly. |
| Both are correct | The two corrections reinforce each other — minimum variance. |
| Both are wrong | No safety net; the estimate will be biased. |

Compare this to the T-Learner, which relies *entirely* on the two outcome models, or a plain inverse-propensity-weighted (IPW) estimator, which relies *entirely* on the propensity model. The DR-Learner requires only one of these to work.

```{admonition} Practical implication
:class: important

When you use the DR-Learner, you do not need to be equally confident in your propensity model and your outcome model. You only need to be confident that at least one of them is reasonable. This is a substantially weaker requirement than alternatives demand.
```

---

## Worked Example: Hotel Flash-Sale Campaign

A hotel sends a flash-sale email to a subset of past guests and measures whether they make a booking. We have already trained the nuisance models and collected their predictions.

### Data

| Guest | Treatment $T$ | Outcome $Y$ | $\hat{\mu}_1(X)$ | $\hat{e}(X)$ |
|:---|:---:|:---:|:---:|:---:|
| **A** — Loyal, high-value | 1 | 1 | 0.80 | 0.90 |
| **B** — Infrequent, low-value | 1 | 1 | 0.40 | 0.10 |
| **C** — New, control group | 0 | 0 | 0.60 | 0.50 |

---

### Step-by-step pseudo-outcome calculation

#### Guest A — The Expected Success

A loyal, high-value guest: the model expected them to book ($\hat{\mu}_1 = 0.80$) and they were very likely to receive the email ($\hat{e} = 0.90$). They were treated and booked.

$$Y^*_A = 0.80 + \frac{1 \times (1 - 0.80)}{0.90} = 0.80 + \frac{0.20}{0.90} \approx 0.80 + 0.22 = \mathbf{1.02}$$

The outcome was well-anticipated. The correction is small. The pseudo-outcome barely moves from the base prediction.

---

#### Guest B — The High-Impact Surprise

An infrequent, low-value guest: the model did not expect them to book ($\hat{\mu}_1 = 0.40$) and they were rarely targeted ($\hat{e} = 0.10$). Yet they were treated and booked.

$$Y^*_B = 0.40 + \frac{1 \times (1 - 0.40)}{0.10} = 0.40 + \frac{0.60}{0.10} = 0.40 + 6.0 = \mathbf{6.40}$$

The outcome was surprising and the individual was rare. The inverse propensity weight ($1/0.10 = 10$) amplifies the residual dramatically. The DR-Learner is signalling: *"Guests like B are rare, but when they respond, it is strong evidence of a large causal effect — pay attention."*

---

#### Guest C — The Control Unit

A new guest who did not receive the email ($T=0$).

$$Y^*_C = 0.60 + \frac{0 \times (0 - 0.60)}{0.50} = 0.60 + 0 = \mathbf{0.60}$$

The correction term vanishes entirely when $T=0$. For untreated individuals, $Y^*$ is simply $\hat{\mu}_1(X)$ — our best prediction of what *would* have happened if they had been treated.

---

### Pseudo-outcome summary

| Guest | $Y$ | $Y^*$ | Interpretation |
|:---|:---:|:---:|:---|
| **A** | 1 | 1.02 | Expected outcome; small adjustment. |
| **B** | 1 | 6.40 | Rare treated unit, surprising outcome — amplified to signal high causal impact. |
| **C** | 0 | 0.60 | Untreated; outcome model fills in the counterfactual. |

The final CATE model $\psi$ is then trained on $(X, Y^*)$. Because Guest B's $Y^*$ is 6.40, the model learns that customers *like* Guest B — low propensity, unexpected responders — are where the real causal lift hides. This is precisely what the double robustness mechanism is designed to surface.

---

## Summary

```{list-table} DR-Learner pipeline at a glance
:header-rows: 1
:widths: 15 25 25 35

* - Phase
  - Inputs
  - Models trained
  - Output
* - **1. Nuisance**
  - $X$, $T$, $Y$
  - $\hat{\mu}_0$, $\hat{\mu}_1$ (outcome) + $\hat{e}$ (propensity)
  - Predicted $Y$ and $P(T=1\mid X)$ for every row.
* - **2. Synthesis**
  - Nuisance predictions
  - DR formula
  - Pseudo-outcomes $Y^*$ — noise-corrected, bias-adjusted targets.
* - **3. Final meta**
  - $X$, $Y^*$
  - Final regressor $\psi$
  - $\hat{\tau}(X)$ — individual-level CATE estimates.
```

The DR-Learner is the preferred meta-learner when:

- Treatment assignment is non-random and confounding is likely.
- You cannot be certain that either the outcome model or the propensity model is perfectly specified.
- You need formal statistical guarantees (doubly robust consistency, semiparametric efficiency).
