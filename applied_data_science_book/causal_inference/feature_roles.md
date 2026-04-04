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

# What Goes Into X? Feature Roles in Causal Inference

In standard supervised learning, a feature either improves predictive accuracy or it doesn't — and the model mostly sorts that out by itself. You can throw in every variable you have and let regularisation handle the rest.

Causal inference is different. **What you include in $X$ determines two independent things:**

1. **Validity** — whether your causal estimates are unbiased at all.
2. **Resolution** — whether your CATE estimates are heterogeneous enough to be actionable.

Including the wrong variables can introduce bias even if the model fits the data perfectly. Omitting the right variables makes your CATE estimates flat and useless for targeting. This page maps out the four distinct roles a feature can play and shows exactly what each one does (and breaks) inside the causal pipeline.

---

## The Four Roles

### 1. Confounders — include for validity

A **confounder** is a variable that affects *both* the probability of being treated *and* the baseline outcome, independently of treatment.

$$Z \text{ is a confounder if: } Z \to T \quad \text{and} \quad Z \to Y$$

**What happens if you omit it:**
The treated and control groups are no longer comparable on $Z$. Any difference in outcomes between them will be partly due to $Z$, not the treatment. Every causal estimator — S-Learner, T-Learner, DR-Learner — will be **biased**. The bias does not shrink with more data; it is structural.

**What happens if you include it:**
The unconfoundedness assumption $(Y(0), Y(1)) \perp T \mid X$ is satisfied for $Z$. The propensity model $\hat{e}(X)$ learns to account for it; the outcome models $\hat{\mu}_0$, $\hat{\mu}_1$ adjust for it in the baseline. Estimates become valid.

**Does including an unnecessary confounder hurt?**
No — if a variable truly is a confounder, including it is always correct. Including a *spurious* confounder (one that is not actually a confounder) adds noise but does not introduce bias, as long as it is not a bad control (see below).

---

### 2. Effect Modifiers — include for resolution

An **effect modifier** (also called a *treatment effect heterogeneity driver*) is a variable along which the *size* of the treatment effect varies.

$$Z \text{ is an effect modifier if: } \tau(x) = E[Y(1)-Y(0) \mid X=x] \text{ differs meaningfully across values of } Z$$

Effect modifiers need not be confounders — they may have nothing to do with who gets treated. Their sole relevance is that $\tau$ is large for some values of $Z$ and small for others.

**What happens if you omit it:**
The final CATE model $\psi(X)$ has no signal to learn the heterogeneity. Its output converges toward a constant — approximately the ATE. You get a valid estimate of the average effect, but **you lose all targeting power**. The whole point of computing CATE (to identify who benefits most) is defeated.

**What happens if you include it:**
The pseudo-outcomes $Y^*$ already encode the true individual-level lift. $\psi(X)$ now has the axis it needs to split on and will produce meaningfully different CATE estimates for different values of $Z$.

**Does including a non-modifier hurt?**
Only in variance. Including a feature that does not modify the treatment effect adds dimensions for $\psi$ to fit noise. With sufficient data this is harmless, but in small samples it can inflate variance.

---

### 3. Regular Predictors — include to reduce variance

A **regular predictor** is a variable that predicts the baseline outcome $Y$ but is unrelated to treatment assignment and does not modify the treatment effect.

$$Z \text{ is a regular predictor if: } Z \to Y \quad \text{but} \quad Z \not\to T \quad \text{and} \quad \tau(x) \text{ is flat in } Z$$

**What happens if you omit it:**
No bias. But the outcome models $\hat{\mu}_0$, $\hat{\mu}_1$ are noisier — they cannot explain part of the variance in $Y$. This increases the variance of the pseudo-outcomes $Y^*$ and, downstream, the variance of the CATE estimates.

**What happens if you include it:**
The outcome models fit better, pseudo-outcomes are less noisy, and the final CATE estimates have lower variance. **Including strong predictors of $Y$ is always a good idea** — even when they are causally irrelevant to the treatment effect — precisely because they tighten the pseudo-outcome.

---

### 4. Bad Controls — do not include

Some variables look like ordinary covariates but their inclusion actively **introduces bias**. There are two main types.

#### Colliders

A **collider** is a variable caused by *both* the treatment and the outcome (or their causes).

$$T \to C \leftarrow Y$$

Conditioning on $C$ (i.e., including it in $X$) opens a spurious statistical association between $T$ and $Y$ that was not present before. This is known as **collider bias** or Berkson's paradox.

*Hotel example:* Suppose `booking_confirmed` is a flag that is set to 1 whenever a guest either received an email ($T=1$) or had a loyalty discount applied (which also drives $Y$). Including `booking_confirmed` in $X$ conditions on a common effect of $T$ and the loyalty pathway, creating a spurious correlation.

#### Mediators

A **mediator** is a variable that lies *on the causal path* from treatment to outcome.

$$T \to M \to Y$$

Including a mediator in $X$ partially blocks the very effect you are trying to measure. The model will attribute some of the treatment effect to changes in $M$ rather than to $T$ itself, and the estimated $\tau$ will be **attenuated** (too small).

*Hotel example:* `email_opened` ($T$ causes the guest to open the email, which causes booking $Y$). If you control for `email_opened`, you are absorbing part of the treatment effect pathway and will underestimate the total effect of sending the email.

```{admonition} Rule of thumb for bad controls
:class: warning

Before adding a variable to $X$, ask: *could treatment $T$ have caused this variable?* If yes, it is a potential mediator or collider — include it only if you have a specific research reason to do so (e.g., estimating a direct effect), and do so deliberately.
```

---

## How Each Role Routes Through the DR Pipeline

The diagram below shows where each feature type exerts its influence inside the Doubly Robust Learner.

```{mermaid}
flowchart LR
    subgraph X["Feature matrix X"]
        C["🔴 Confounder\ne.g. loyalty tier"]
        M["🟢 Effect modifier\ne.g. age"]
        P["🔵 Regular predictor\ne.g. prev_bookings"]
    end

    subgraph N["Phase 1 — Nuisance models"]
        EM["Outcome models\nμ̂₀(X), μ̂₁(X)"]
        PM["Propensity model\nê(X)"]
    end

    PO["Phase 2\nPseudo-outcome Y*"]
    CATE["Phase 3\nFinal CATE model ψ(X)"]

    C -->|"removes selection bias"| PM
    C -->|"adjusts baseline"| EM
    M -->|"minor role"| EM
    M -->|"minor role"| PM
    P -->|"explains variance in Y"| EM
    EM --> PO
    PM --> PO
    PO --> CATE
    M --->|"main signal\nfor heterogeneity"| CATE

    style C fill:#f8d7da,stroke:#721c24,color:#000
    style M fill:#d4edda,stroke:#28a745,color:#000
    style P fill:#cce5ff,stroke:#004085,color:#000
    style CATE fill:#fff3cd,stroke:#856404,color:#000
```

**Reading the diagram:**

- **Confounders (red)** flow into *both* the propensity model and the outcome models. They are doing two jobs: correcting selection bias in $\hat{e}$ and adjusting the baseline in $\hat{\mu}$.
- **Effect modifiers (green)** pass through the nuisance models as ordinary features, but their real payoff is in Phase 3: the final CATE model $\psi$ splits on them to produce heterogeneous predictions.
- **Regular predictors (blue)** primarily improve the outcome models, tightening $Y^*$. They have no direct influence on the final CATE surface.

---

## Toy Example: Hotel Flash-Sale Campaign

A hotel is estimating the CATE of a flash-sale email ($T$) on bookings ($Y$). Four features are available.

### Features

| Feature | Variable | Type |
|:---|:---|:---|
| Loyalty tier (Gold / Silver / None) | `loyalty` | Confounder |
| Previous bookings (count) | `prev_bookings` | Regular predictor |
| Age | `age` | Effect modifier |
| On VIP marketing list (0 / 1) | `vip` | Confounder **and** effect modifier |

---

### `loyalty` — Confounder

The hotel's marketing team specifically targets Gold members with the email, so `loyalty` affects $T$. Gold members also have higher baseline booking rates regardless of any email, so `loyalty` affects $Y(0)$.

- **Omit it:** Gold members dominate the treated group. Their higher baseline booking rate is wrongly attributed to the email. All meta-learner estimates are upward-biased.
- **Include it:** The propensity model learns that Gold members have high $\hat{e}$, so their treated outcomes are down-weighted appropriately. Bias is removed.
- **Does it modify the effect?** Possibly — but that is a secondary question. Its primary job is deconfounding.

---

### `prev_bookings` — Regular predictor

Emails are sent independently of booking history (the campaign targets by loyalty tier, not recency). But guests with many previous bookings are generally more likely to book again. The treatment effect (the *extra* lift from the email) is roughly constant across booking history.

- **Omit it:** No bias. But outcome models explain less variance in $Y$, pseudo-outcomes are noisier, CATE estimates have higher variance.
- **Include it:** Outcome models fit better. Cleaner pseudo-outcomes. Worth including even though it is causally inert with respect to the treatment effect.

---

### `age` — Effect modifier

The email campaign is sent without age-targeting — older and younger guests are equally likely to receive it, so `age` is not a confounder. But younger guests are price-sensitive and respond strongly to flash sales; older guests are habit-driven and book based on prior experience regardless of promotions.

- **Omit it:** No bias (not a confounder). But $\psi(X)$ cannot distinguish young from old guests. It returns a single average lift for everyone. Targeting becomes impossible.
- **Include it:** $\psi$ learns that young guests have high CATE and older guests have low CATE. The hotel can direct the campaign budget toward the age group where it creates real value.

---

### `vip` — Confounder and effect modifier

Being on the VIP list both increases the probability of receiving the email (marketing targets VIPs) *and* increases baseline bookings (VIPs are high-value, engaged customers). So `vip` is a confounder. But VIPs also happen to respond more strongly to personalised flash sales than non-VIPs — so `vip` also modifies the treatment effect.

- **Omit it:** Bias (confounding) *plus* lost heterogeneity. Double penalty.
- **Include it:** Deconfounds the estimate *and* gives $\psi$ a strong axis of CATE variation to exploit. Including it is unambiguously correct.

---

### Summary table

| Feature | Role | Affects $T$? | Affects baseline $Y$? | Modifies $\tau$? | Consequence of omitting |
|:---|:---|:---:|:---:|:---:|:---|
| `loyalty` | **Confounder** | Yes | Yes | Weakly | **Bias** in all estimates |
| `prev_bookings` | **Regular predictor** | No | Yes | No | Higher variance, no bias |
| `age` | **Effect modifier** | No | Weakly | **Yes** | Flat CATE, no targeting power |
| `vip` | **Both** | Yes | Yes | **Yes** | **Bias** + flat CATE |

---

## The Three Questions

Before adding any variable to $X$, ask:

```{list-table}
:header-rows: 1
:widths: 5 40 55

* - #
  - Question
  - If yes →
* - 1
  - Does this variable affect who gets treated?
  - Include it (confounder — needed for validity).
* - 2
  - Does this variable change how *much* the treatment helps?
  - Include it (effect modifier — needed for CATE resolution).
* - 3
  - Could the treatment have *caused* this variable?
  - Do **not** include it without careful thought (potential mediator or collider).
```

If the answer to all three is "no", the variable is a regular predictor — worth including to reduce variance, but not critical for validity or resolution.
