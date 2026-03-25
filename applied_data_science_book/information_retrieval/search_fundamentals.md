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

# Search Fundamentals

Classical information retrieval provides the vocabulary and baseline algorithms
that more sophisticated approaches are measured against. Even when you end up
deploying a neural ranker, understanding BM25 and inverted indexes makes you a
far better engineer of the overall pipeline.

## Core concepts

### The inverted index

An **inverted index** maps every token in a corpus to the list of documents that
contain it (the *posting list*). It is the data structure behind every full-text
search engine, from Lucene/Elasticsearch to SQLite FTS.

```
token → [(doc_id, term_freq), ...]
```

At query time the engine intersects posting lists for all query tokens, scores
each candidate, and returns the top-k results.

### TF-IDF

**Term Frequency – Inverse Document Frequency** scores how important a term `t`
is to a document `d` within a corpus `D`:

$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$

- $\text{TF}(t,d)$: frequency of `t` in `d` (often log-normalised).
- $\text{IDF}(t,D) = \log \frac{|D|}{|\{d \in D : t \in d\}|}$: penalises terms
  that appear in many documents.

### BM25

**BM25** (Best Match 25) is the dominant sparse retrieval function and the
default in Elasticsearch / OpenSearch. It extends TF-IDF with two tunable
parameters:

- **k1** (typical range 1.2–2.0): controls term-frequency saturation.
- **b** (typical 0.75): controls document-length normalisation.

$$
\text{BM25}(t, d) = \text{IDF}(t) \cdot
  \frac{f(t,d) \cdot (k_1 + 1)}
       {f(t,d) + k_1 \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

BM25 is a strong baseline that is surprisingly hard to beat without significant
training data.

## Evaluation metrics

| Metric | Description |
|---|---|
| **Precision@k** | Fraction of top-k results that are relevant |
| **Recall@k** | Fraction of all relevant docs found in top-k |
| **MRR** | Mean Reciprocal Rank — rewards finding the first relevant result early |
| **MAP** | Mean Average Precision — area under the precision-recall curve, averaged over queries |
| **NDCG@k** | Normalised Discounted Cumulative Gain — handles graded relevance; the standard metric in LTR benchmarks |

## Query expansion

The fundamental problem with keyword search is **vocabulary mismatch**: a user
searches for *automobile* but the document only contains the word *car*. Query
expansion addresses this by enriching the query with semantically related terms
before passing it to the retrieval engine.

### Embedding-based query expansion

Rather than replacing the inverted index with dense retrieval, embedding-based
expansion places an intelligent **rewriting layer** in front of your existing
BM25 engine. The pipeline has four steps:

**1. Build a global vocabulary (offline)**

An unsupervised model (Word2Vec, GloVe, FastText) is trained on a large corpus.
Words that appear in similar contexts end up physically close in the
high-dimensional vector space — *hotel*, *motel*, and *inn* cluster together
because they occur in near-identical sentence structures.

**2. Generate expansion candidates (at query time)**

For each query term, find its $k$-nearest neighbours in embedding space using
cosine similarity:

$$
\text{sim}(\vec{q}, \vec{c}) = \frac{\vec{q} \cdot \vec{c}}{\|\vec{q}\| \|\vec{c}\|}
$$

For the query term *laptop*, this surfaces candidates like *notebook*,
*macbook*, and *pc*.

**3. Weight the expanded query**

The original terms carry the highest weight; expansion terms are discounted by
their similarity score to avoid diluting the original intent:

```
Original query:  laptop
Expanded query:  laptop^1.0 OR notebook^0.85 OR macbook^0.72
```

**4. Execute lexical retrieval**

The weighted expanded query is passed unchanged to BM25. Because the query now
contains synonyms, it retrieves documents that never explicitly mentioned
*laptop*.

### Advantages over dense retrieval

| Property | Embedding expansion | Dense retrieval |
|---|---|---|
| Infrastructure | Reuses existing inverted index | Requires vector database |
| Explainability | Can inspect the expanded query string | Black-box vector similarity |
| Index cost | Documents indexed as plain text | Every document needs a dense vector |
| Latency | Near-zero overhead on retrieval | ANN search adds ~1–10 ms |

### Query drift

The principal risk is **query drift** (semantic drift): unsupervised embeddings
capture *relatedness*, not *synonymy*. The word *hot* sits close to *cold* in
most embedding spaces because both are temperatures used in identical sentence
patterns. A naive expansion of *"hot weather destinations"* may inject *cold*
and return entirely wrong results.

Common mitigations:

- **Similarity threshold** — only accept candidates above a cosine similarity
  cutoff (e.g. 0.7), discarding loosely related terms.
- **Part-of-speech filtering** — restrict expansion to the same grammatical
  category as the original term.
- **Supervised re-scoring** — use a small set of labelled query–document pairs
  to learn which expansions actually improve recall without hurting precision.
- **Constrained expansion** — expand only within a domain-specific vocabulary
  (e.g. product catalogue terms) rather than a general-purpose embedding space.

### When to use query expansion vs. dense retrieval

Query expansion is a good fit when:
- You have an existing BM25 / Elasticsearch stack and cannot afford to rebuild
  around a vector database.
- Explainability matters (support, compliance, debugging).
- Your document corpus changes frequently (dense vectors need re-encoding).

Prefer dense retrieval when:
- You have enough compute to embed and index the full corpus.
- Vocabulary mismatch is severe and diverse (e.g. multilingual queries,
  colloquial language, long-tail queries).
- You are building a RAG pipeline where recall quality is paramount.

## Practical notes

- BM25 is the right starting point for any new search project.
- Tune `k1` and `b` on a labelled query set before reaching for dense retrieval.
- Use NDCG@10 as your primary offline metric; it correlates well with online
  A/B test outcomes.
- Add embedding-based query expansion as a low-infrastructure improvement step
  between vanilla BM25 and full dense retrieval.
