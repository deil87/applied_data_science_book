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

# Information Retrieval, Search and Ranking

Information retrieval (IR) is the science of finding relevant material — usually
documents or data records — from a large collection in response to a query. It
underpins virtually every product that helps users find things: web search
engines, e-commerce product search, recommendation feeds, enterprise knowledge
bases, and code search tools.

## Why it matters in Applied Data Science

Modern data science teams encounter IR problems constantly:

- **Internal search** — employees querying document stores, wikis, or data
  catalogs.
- **Customer-facing search** — product discovery, content recommendation,
  support ticket routing.
- **RAG pipelines** — retrieval-augmented generation systems that fetch context
  before passing it to a language model.
- **Ad targeting & ranking** — selecting and ordering items from a candidate
  pool to maximise a business objective.

Getting retrieval right is often more impactful than tuning the downstream model,
because garbage-in ⟹ garbage-out applies at the retrieval stage too.

## Chapter structure

This chapter is organised into three sections:

| Section | Topics |
|---|---|
| [Search Fundamentals](search_fundamentals) | Inverted indexes, TF-IDF, BM25, evaluation metrics (NDCG, MAP, MRR) |
| [Semantic Search](semantic_search) | Dense retrieval, embeddings, approximate nearest-neighbour search, vector databases |
| [Learning to Rank](learning_to_rank) | Pointwise / pairwise / listwise LTR, LambdaMART, neural rankers, custom loss functions |

Each section combines conceptual explanations with practical code examples.
Resource articles and notebooks will be added to the `information_retrieval/`
subfolder and synthesised into the relevant section pages.
