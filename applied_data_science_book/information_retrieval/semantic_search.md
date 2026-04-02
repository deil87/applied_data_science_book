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

# Semantic Search

Sparse retrieval (BM25) relies on exact token overlap between query and document.
Semantic search addresses the **vocabulary mismatch problem** by mapping both
queries and documents into a shared dense vector space where proximity reflects
meaning rather than surface form.

## From sparse to dense retrieval

| | Sparse (BM25) | Dense (bi-encoder) |
|---|---|---|
| Representation | Bag-of-words vector | Fixed-size embedding |
| Index size | Proportional to vocabulary | Proportional to corpus size |
| Latency | Sub-millisecond | ~1–10 ms with ANN |
| Handles synonyms | No | Yes |
| Training data needed | No | Yes (or pre-trained model) |
| Explainability | High | Low |

In practice the two are complementary: **hybrid search** (BM25 + dense) often
outperforms either alone.

## Distributional semantics: topical vs. typical relatedness

Before picking an embedding model, it helps to understand the two fundamentally
different ways words can be "semantically close" — because different model
architectures optimise for different kinds of closeness.

### Two types of relatedness

**Topical (associative) relatedness** — words belong to the same *subject
matter* or semantic field. They co-occur frequently in the same *documents*, but
one cannot necessarily substitute for the other in a sentence.

> Examples: *doctor* & *hospital*, *coffee* & *mug*, *sun* & *beach*

**Typical (substitutionary / functional) relatedness** — words share semantic
features and function similarly in a sentence. They can often replace each other
without changing grammatical structure.

> Examples: *car* & *bus*, *big* & *huge*, *run* & *walk*

```{image} images/topical_vs_typical.png
:alt: Diagram showing topical vs. typical semantic relatedness
:width: 80%
:align: center
```

### How classic embedding models map onto this distinction

| Model | Mechanism | Prioritises |
|---|---|---|
| **Word2Vec** (Skip-gram / CBOW) | Local sliding window (~5 words); predicts nearby context words | **Typical** — learns words that appear next to the same context words |
| **LSA** (Latent Semantic Analysis) | SVD on a global term–document co-occurrence matrix | **Topical** — learns words that appear in the same overall documents |
| **GloVe** | Factorises a global *word–word* co-occurrence matrix | **Both** — strong on analogies (typical) while still capturing topic (topical) |

This distinction has direct practical consequences:

- **Query expansion** built on Word2Vec will surface good *synonyms* (*laptop →
  notebook*) but may miss thematically related terms (*laptop → charger*).
- **LSA-style models** are better at clustering documents by topic, but poorer
  at synonym detection.
- **GloVe** is a safer default when you want a single static embedding for
  general-purpose retrieval.
- **Contextual models** (BERT, Sentence-Transformers) learn both types
  simultaneously from billions of tokens, which is a key reason they outperform
  all static embeddings on retrieval benchmarks.

## Text embeddings

A **bi-encoder** independently embeds the query and each document with the same
encoder (e.g. a fine-tuned BERT / Sentence-Transformers model). Similarity is
computed as the dot product or cosine of the two vectors.

Popular pre-trained models:
- `all-MiniLM-L6-v2` — fast, good general-purpose baseline (384 dims)
- `text-embedding-3-small` — OpenAI, strong out-of-the-box
- `e5-large-v2` — strong on BEIR benchmark
- `bge-m3` — multilingual, supports sparse + dense hybrid natively

## Approximate Nearest Neighbour (ANN) search

Brute-force cosine search over millions of vectors is too slow. ANN indexes
trade a small amount of recall for orders-of-magnitude speed-up:

| Algorithm | Library | Notes |
|---|---|---|
| HNSW | `hnswlib`, Faiss, Weaviate | Best recall/speed trade-off; in-memory |
| IVF-PQ | Faiss | Compressed; scales to billions of vectors |
| ScaNN | Google ScaNN | Optimised for Google-scale workloads |
| DiskANN | Microsoft | SSD-based; low memory footprint |

## Vector databases

Managed ANN search + metadata filtering + CRUD:

- **Pinecone** — fully managed, serverless option
- **Weaviate** — open-source, built-in hybrid search
- **Qdrant** — open-source, Rust-based, fast filtering
- **Milvus / Zilliz** — open-source, GPU-accelerated
- **pgvector** — vector extension for PostgreSQL; simplest ops story

## Retrieval-Augmented Generation (RAG)

Semantic search is the retrieval backbone of RAG systems:

```
query → embed → ANN search → top-k chunks → LLM prompt → answer
```

Retrieval quality has an outsized impact on answer quality — improving the
retriever is usually more effective than prompt engineering.

## Evaluation

Use the same NDCG / MRR / Recall@k metrics as for sparse retrieval. The
[BEIR benchmark](https://github.com/beir-cellar/beir) provides a standard
heterogeneous test suite across 18 retrieval tasks.


