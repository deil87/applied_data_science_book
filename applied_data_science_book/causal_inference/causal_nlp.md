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

# Causal Inference in NLP and LLMs

## Why LLMs Need Causality

Standard large language models are trained to minimise next-token prediction loss. This is a fundamentally **associative** objective: the model learns that certain words co-occur with certain outcomes and compresses that statistical pattern into its weights. It is extraordinarily good at this — but it has a hard ceiling.

When a business question shifts from *"What is likely to happen?"* to *"Why did this happen?"* or *"What would happen if we changed X?"*, pure association breaks down. LLMs trained on correlational data will:

- Mistake co-occurrence for causation (umbrellas appear with rain → umbrellas cause rain).
- Hallucinate plausible-sounding causal explanations that have no grounding in the data-generating process.
- Fail to answer counterfactual questions reliably ("Would this customer have churned if we had offered a discount?").

The intersection of LLMs and causal inference — often called **Causal NLP** — addresses this ceiling. It is currently one of the most active frontiers in applied AI research, because it is also where the most business value is locked up.

---

## The Four Use-Case Families

There are two directions in which LLMs and causal methods interact, producing four distinct use-case families.

```{mermaid}
flowchart LR
    subgraph LLM["LLM capabilities"]
        EXT["Text understanding\nand generation"]
    end

    subgraph CI["Causal Inference methods"]
        EST["Causal estimators\nMeta-learners, DML, DiD"]
    end

    subgraph UC1["① Extraction"]
        U1["Mine causal variables\nfrom unstructured text"]
    end

    subgraph UC2["② Counterfactuals"]
        U2["Generate synthetic\ncounterfactual text"]
    end

    subgraph UC3["③ Causal reasoning"]
        U3["Fine-tune LLMs to\nreason causally"]
    end

    subgraph UC4["④ ROI evaluation"]
        U4["Measure business impact\nof deployed LLM features"]
    end

    LLM -->|"feeds structured\nvariables into"| UC1
    UC1 --> CI

    LLM -->|"creates paired\ndatasets for"| UC2
    UC2 --> CI

    CI -->|"causal datasets\nand penalties shape"| UC3
    UC3 --> LLM

    CI -->|"HTE / DiD measures\nvalue of"| UC4
    UC4 --> LLM

    style LLM fill:#cce5ff,stroke:#004085,color:#000
    style CI fill:#d4edda,stroke:#28a745,color:#000
    style UC1 fill:#f8f9fa,stroke:#6c757d,color:#000
    style UC2 fill:#f8f9fa,stroke:#6c757d,color:#000
    style UC3 fill:#f8f9fa,stroke:#6c757d,color:#000
    style UC4 fill:#f8f9fa,stroke:#6c757d,color:#000
```

---

### Use Case 1 — Causal Information Extraction (LLM → Causal Inference)

Traditional causal inference operates on tabular data: pricing columns, click flags, demographic integers. But enormous quantities of causal information are trapped in unstructured text — support tickets, medical notes, earnings call transcripts, product reviews.

**The problem:** You want to build a causal graph explaining why customers churn, but the real reasons exist only in free-text chat logs. You cannot run a meta-learner on raw prose.

**The solution:** Fine-tune a smaller, cost-effective LLM (Llama 3 8B, Mistral 7B) specifically for **causal information extraction**. The model is trained to read a transcript and output structured variables:

```
Input:  "The checkout kept freezing and I lost my cart three times. I'm done."
Output: { cause: "repeated checkout failures", effect: "cancellation intent", sentiment: "frustrated" }
```

These structured outputs become rows in a tabular dataset that is fed into a standard causal estimator — a DR-Learner, a DML model, a Difference-in-Differences setup.

```{admonition} Why fine-tune rather than prompt?
:class: tip

Zero-shot or few-shot prompting of a large general model extracts some causal structure, but is inconsistent across edge cases and hallucination-prone on domain-specific jargon. A fine-tuned smaller model is cheaper per token, more consistent, and easier to validate on a labelled holdout set.
```

---

### Use Case 2 — Counterfactual Text Generation (LLM augments Causal Data)

To measure a causal effect, you need the counterfactual: what *would* have happened if the treatment had been different? In observational text data you only ever see one version of each message.

**The problem:** You want to know the causal impact of *email tone* (urgent vs. friendly) on booking conversion, but every historical email was written in exactly one tone per customer.

**The solution:** Use an LLM to rewrite each historical email in the alternative tone, producing synthetic paired datasets:

```
Original (urgent):   "Last chance — only 2 rooms left at this price!"
Synthetic (friendly): "We wanted to let you know a great rate is still available for your dates."
```

By running causal estimators on these paired datasets, you can estimate the treatment effect of tone independently of all the other variables in the email.

This use case is covered in depth in the next two sections, because it raises a non-trivial question: *how do you validate the synthetic text, and which algorithm do you trust to estimate the effect?*

---

### Use Case 3 — Fine-Tuning for Causal Reasoning (Causal Inference → LLM)

Standard LLMs hallucinate because they confuse statistical co-occurrence with causation. In high-stakes domains — healthcare, fintech, legal — a correlational hallucination is unacceptable.

**The problem:** An AI clinical assistant suggests a drug because it has seen the drug name and the diagnosis co-occur frequently in training data, not because of a verified causal pathway.

**The solution:** Two complementary approaches:

- **Causal dataset fine-tuning:** Train the LLM on curated (premise, causal conclusion) pairs where the conclusion is verified by a Structural Causal Model (SCM). The model learns to distinguish causal from correlational statements.
- **SCM-penalised reinforcement learning:** During RL fine-tuning (RLHF or similar), add a penalty whenever the model's output violates a known causal constraint encoded in a domain SCM. The reward function becomes causally informed.

The result is a model that generates text anchored to the causal mechanics of the domain rather than to surface-level co-occurrence statistics.

---

### Use Case 4 — Evaluating the ROI of LLM Features (Causal Inference evaluates LLM)

When you deploy a GenAI feature — a RAG-powered chatbot, an automated summariser, an AI writing assistant — you need to prove it *caused* a business metric improvement, not just that the metric improved at the same time.

**The problem:** You launch an LLM-powered copilot for your customer support team. Resolution times drop 18% in the following month. Was it the copilot, or is January just a slower month for support tickets?

**The solution:** Standard causal methods applied to the rollout:

- **Difference-in-Differences (DiD):** Compare the before/after trend in resolution times for the team using the copilot against a control team that has not yet received access. The difference of differences isolates the copilot's effect from the seasonal trend.
- **Heterogeneous Treatment Effects (HTE):** Use a meta-learner (e.g., the DR-Learner from {doc}`doubly_robust`) to identify *which types* of agents benefit most — senior agents with complex tickets, or junior agents with simple ones? This informs targeted rollout and training priorities.

---

## Counterfactual Text Generation — Deep Dive

Because Use Case 2 requires careful validation before any causal estimator is run, we expand it here into two explicit phases.

### Phase 1 — Evaluating the Synthetic Text

Before running any causal mathematics, you must prove the LLM did not "cheat." If you asked it to change the tone to *urgent*, you need to confirm it did not also silently add a fake 20% discount (which would be a massive confounder). Three quality-control metrics gate the synthetic data:

| Metric | Method | Target threshold | What failure looks like |
|:---|:---|:---:|:---|
| **Treatment adherence** | Fine-tune a DistilBERT classifier on tone labels; score synthetic text | >90% probability for target tone | LLM made the text "slightly more formal" but not actually urgent |
| **Semantic preservation** | SentenceTransformers cosine similarity between original and synthetic embeddings | >0.85 | LLM added a discount, changed the product, or hallucinated facts |
| **Fluency** | GPT-2 perplexity on synthetic text vs. originals | No significant spike | LLM produced stilted or repetitive phrasing that users would not respond to naturally |

Reject synthetic pairs that fail any gate and flag them for manual review or regeneration with a stricter prompt.

---

### Phase 2 — Causal Algorithms for Effect Estimation

Once validated pairs $(X_{\text{original}}, X_{\text{synthetic}})$ exist and historical outcomes $Y_{\text{observed}}$ are available for the original text, three algorithms are applicable, in increasing order of complexity.

#### Algorithm 1 — T-Learner with Text Embeddings

The most practical starting point. Treat the causal estimation as a supervised regression problem.

1. Embed all historical emails using a text embedding model (`text-embedding-3-small`, `all-MiniLM-L6-v2`, etc.).
2. Train an outcome model $\hat{\mu}$ — typically XGBoost or a shallow neural network — to predict $Y$ (conversion, click, booking) from the embedding.
3. Embed the synthetic (rewritten) emails.
4. Pass the synthetic embeddings through $\hat{\mu}$ to get the predicted counterfactual outcome $\hat{Y}_{\text{synthetic}}$.
5. Compute the Individual Treatment Effect:

$$\text{ITE}_i = Y^{\text{observed}}_i - \hat{Y}^{\text{synthetic}}_i$$

**Strengths:** Simple, explainable, fast to iterate.  
**Weakness:** The outcome model must generalise from real to synthetic embeddings — if the synthetic text sits in a different region of the embedding space, extrapolation error inflates the ITE estimates.

---

#### Algorithm 2 — Double Machine Learning (DML)

The gold standard when the text contains heavy confounding — for example, angry customers write longer emails, and longer emails naturally take longer to resolve regardless of tone.

DML trains two residual models to strip the confounding signal out before estimating the treatment effect. **However**, applying DML naively to raw text embeddings breaks — this is addressed in full in the next section.

When applied correctly (using a neutralised text representation — see {ref}`dml-text-problem`), DML works as follows:

1. Represent the confounders $W$ separately from the treatment $T$ (tone).
2. **Model A:** Predict $Y$ from $W$ only, get residual $\tilde{Y} = Y - \hat{Y}(W)$.
3. **Model B:** Predict $T$ from $W$ only, get residual $\tilde{T} = T - \hat{T}(W)$.
4. Regress $\tilde{Y}$ on $\tilde{T}$. The coefficient is the causal effect of tone, purged of all confounding from email length, topic, and vocabulary.

**Strengths:** Formally correct under mild assumptions; handles rich confounding.  
**Weakness:** Requires that $T$ and $W$ be separable — which is non-trivial when both live inside the same text (see next section).

---

#### Algorithm 3 — Propensity Score Matching via Embeddings

When you do not trust a predictive model to extrapolate to synthetic text, avoid extrapolation entirely by finding real historical counterparts.

1. Embed the synthetic email.
2. Search your historical database for the nearest-neighbour real email in embedding space — one that naturally looks like the synthetic version but received the alternative treatment in the real world.
3. Compare their real observed outcomes directly.

**Strengths:** No model extrapolation; uses only real outcomes.  
**Weakness:** Matching quality degrades in high-dimensional embedding spaces and breaks down if the alternative treatment was rarely observed historically (sparse overlap).

---

#### Practical Recommendation

| Scenario | Recommended approach |
|:---|:---|
| Quick prototype, binary outcome (click / no click) | T-Learner + `text-embedding-3-small` + XGBoost |
| Heavy confounding (topic, length, user history) | DML + LLM concept extraction to separate $W$ from $T$ |
| Distrust of synthetic-to-real extrapolation | Propensity score matching in embedding space |
| High-stakes domain (healthcare, fintech) | SCM-penalised fine-tuning + DML with adversarial disentanglement |

---

(dml-text-problem)=
## The DML-with-Text Problem

You may have spotted a tension in Algorithm 2 above: if the email text is embedded as a single dense vector, how can Model A "ignore the tone" when the tone is *baked into that vector*? The answer is that it cannot — and trying anyway causes DML to fail silently.

### The Positivity Violation

Causal inference requires **overlap** (positivity): for any given set of confounders $W = w$, there must be a non-zero probability of receiving either treatment.

$$0 < P(T=1 \mid W=w) < 1 \quad \forall\, w$$

When you feed a raw full-text embedding into DML, Model B (the propensity model) is trying to predict tone from an embedding that *already encodes the tone perfectly*. It achieves near-100% accuracy. This means:

- For every embedding, the model says the person had a ~99.9% chance of being in the angry group.
- There is no overlap — the probability is never in $(0, 1)$.
- DML's residual $\tilde{T}$ collapses toward zero, and the final regression produces a meaningless estimate.

```{mermaid}
flowchart TB
    subgraph WRONG["Raw embedding approach — DML breaks"]
        direction LR
        RAW["Raw email embedding\ne.g. OpenAI text-embedding-3"]
        TONE["Tone signal\n(= Treatment T)"]
        CONF["Topic, length, vocabulary\n(= Confounders W)"]
        RAW -->|"inseparably encodes"| TONE
        RAW -->|"inseparably encodes"| CONF
        MODELA["Model A\n(predict Y from embedding)"]
        MODELB["Model B\n(predict T from embedding)"]
        RAW --> MODELA
        RAW --> MODELB
        MODELB -->|"near 100% accuracy\n→ positivity collapses"| FAIL["DML estimate\nis meaningless"]
    end

    subgraph RIGHT["Neutralised representation — DML works"]
        direction LR
        NEUT["Neutralised text\n(tone stripped out)"]
        W["Confounder embedding W\n(topic, length, facts only)"]
        NEUT --> W
        MA["Model A\npredict Y from W"]
        MB["Model B\npredict T from W"]
        W --> MA
        W --> MB
        MB -->|"uncertain prediction\n→ positivity holds"| OK["DML estimate\nis valid"]
    end

    style FAIL fill:#f8d7da,stroke:#721c24,color:#000
    style OK fill:#d4edda,stroke:#28a745,color:#000
    style WRONG fill:#fff5f5,stroke:#dc3545,color:#000
    style RIGHT fill:#f5fff5,stroke:#28a745,color:#000
```

### Three Solutions

#### Solution 1 — LLM Concept Extraction (recommended)

Use an LLM as a preprocessing filter *before* any embedding is computed. Prompt it to extract only the factual, emotionally neutral content of the email:

> *"Rewrite the following email keeping only the factual business context — the specific problem, the product mentioned, and the account history. Remove all emotional language, adjectives conveying urgency or frustration, and any tone markers."*

| Version | Text |
|:---|:---|
| **Raw** | *"Fix this stupid bug immediately — your platform is garbage and I'm losing money every hour!"* |
| **Extracted $W$** | *"User reports a software bug causing financial loss. Requests immediate resolution."* |

Embed the extracted text. This embedding is your confounder $W$. Feed it into both Model A and Model B. Model B can now only predict tone from topic and severity signals — not from the tone itself — so its accuracy is well below 100% and positivity is preserved.

**Cost:** One LLM call per historical email. No custom training required.

---

#### Solution 2 — Adversarial Disentanglement

If you are training your own embedding model (a custom RoBERTa or domain-specific encoder), add an adversarial head during training:

1. The main encoder produces a representation $h$ of the text.
2. A secondary classifier tries to predict the tone label $T$ from $h$.
3. The encoder is trained to maximise task performance (e.g., outcome prediction) *and* to minimise the secondary classifier's accuracy on $T$.

Over training, the encoder learns to produce embeddings that capture topic and length well but mathematically cannot be used to recover the tone. The result is a tone-agnostic vector that satisfies positivity by construction.

**Cost:** Significant — requires labelled tone data, custom training infrastructure, and careful hyperparameter tuning. Scientifically the most rigorous option.

---

#### Solution 3 — Pre-Treatment Text

The structural solution: change *which* text you embed, choosing text that predates the treatment expression.

Instead of embedding the email the customer just sent (which contains today's tone), embed the **previous $k$ emails** in the customer's conversation history:

- Past emails predict whether the customer is likely to convert ($Y$) — customers with rich prior engagement book more.
- Past emails predict whether the customer is likely to write an angry email today ($T$) — customers with prior complaints are more likely to be frustrated again.
- But past emails do not *contain* today's treatment, so positivity is preserved by construction.

**Cost:** Near zero — just a change in which rows you aggregate before embedding. Works whenever sufficient conversation history exists.

```{admonition} Which solution to choose?
:class: important

Start with **Solution 1** (LLM concept extraction). It requires no custom training, is cheap to run, and is easy to audit — you can manually inspect the extracted $W$ texts to verify that tone has been removed. Move to Solution 3 if conversation history is available and rich. Use Solution 2 only if you are building a long-lived production system that justifies the training investment.
```

---

## Summary

Causal NLP sits at the intersection of two fields that were largely separate until recently. The four use-case families cover the main ways they reinforce each other:

| Use case | Direction | Core method |
|:---|:---|:---|
| Causal information extraction | LLM → Causal | Fine-tuned extraction model feeds tabular estimator |
| Counterfactual text generation | LLM augments Causal | LLM rewrites + Phase 1 QC + Phase 2 estimator |
| Causal reasoning fine-tuning | Causal → LLM | SCM-constrained training / RL penalty |
| LLM feature ROI evaluation | Causal evaluates LLM | DiD, HTE with DR-Learner |

The deepest technical challenge — the DML-with-text positivity violation — arises whenever the treatment is expressed *within* the text being embedded. The fix is always a form of **representation separation**: ensure the embedding fed into the causal estimator encodes only confounders $W$, not the treatment $T$ itself. LLM concept extraction is the most accessible way to achieve this in practice.
