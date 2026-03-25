# AGENTS.md — Project Meta-Information

This file provides context for AI coding agents (and human contributors) about the
structure, conventions, and workflow of this repository.

---

## Project Overview

**Title:** Applied Data Science - Diary of a Lonely Scientist
**Author:** Andrey Spiridonov
**Tech stack:** [Jupyter Book](https://jupyterbook.org) v1.0.4, MyST Markdown, Jupyter Notebooks
**Live site:** https://deil87.github.io/applied_data_science_book/intro.html
**Repository:** https://github.com/deil87/applied_data_science_book

The book covers applied data science topics across various industry domains. Each
chapter focuses on a real-world problem area (fraud detection, marketing, trading,
etc.) and contains one or more sub-sections that dive into specific techniques or
use cases.

---

## Repository Layout

```
applied_ds_book/               ← repo root
├── AGENTS.md                  ← this file
├── README.md
├── adsb_env/                  ← Python virtual environment (not committed)
└── applied_data_science_book/ ← Jupyter Book source root
    ├── _config.yml            ← book-level configuration
    ├── _toc.yml               ← table of contents (single source of truth for structure)
    ├── _static/custom.css     ← custom CSS
    ├── _build/                ← build output (git-ignored)
    ├── intro.md               ← book landing page (TOC root)
    ├── references.bib         ← BibTeX bibliography
    ├── requirements.txt       ← all Python dependencies for the book
    ├── <chapter>.md           ← flat chapter index files (older chapters)
    ├── <chapter>_<section>.md ← flat section files (older chapters)
    └── <chapter>/             ← subfolder-per-chapter (newer pattern, see below)
        ├── index.md           ← chapter landing page
        └── <section>.md      ← section files
```

---

## File & Naming Conventions

### Older chapters (flat layout)
All files live directly in `applied_data_science_book/`:
- Chapter index: `<topic>.md` (e.g. `marketing.md`)
- Section files: `<topic>_<subtopic>.md` (e.g. `marketing_attribution.md`)
- Snake_case throughout; underscores as word separators.

### Newer chapters (subfolder layout)
Each chapter gets its own subfolder:
- `<topic>/index.md` — chapter landing page
- `<topic>/<subtopic>.md` — section files

The `information_retrieval/` chapter follows this subfolder pattern.

### File types
- `.md` — MyST Markdown (preferred for prose-heavy content)
- `.ipynb` — Jupyter Notebook (preferred when code execution is central)

### MyST Markdown front matter (required for `.md` files)
```yaml
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
```

---

## Table of Contents (`_toc.yml`)

Format is `jb-book` with `root: intro`. Chapters listed under `chapters:`;
sub-pages use `sections:` under a parent `file:`.

- Paths are relative to `applied_data_science_book/`, **without file extensions**.
- For subfolder chapters use `file: <folder>/index` as the parent.
- Example entry for a subfolder chapter:

```yaml
- file: information_retrieval/index
  sections:
    - file: information_retrieval/search_fundamentals
    - file: information_retrieval/semantic_search
    - file: information_retrieval/learning_to_rank
```

---

## Building Locally

```bash
# 1. Activate the virtual environment (from repo root)
source adsb_env/bin/activate

# 2. Install / update dependencies
pip install -r applied_data_science_book/requirements.txt

# 3. Build the book (from repo root)
jupyter-book build applied_data_science_book/

# 4. Open in browser
open applied_data_science_book/_build/html/index.html
```

> **Known issue:** The legacy `Learning to rank/` folder (space in name) is
> automatically picked up by Jupyter Book and generates build warnings. It can
> safely be ignored until it is renamed or integrated into the TOC properly.

To do a **clean rebuild** (clears cached outputs):
```bash
jupyter-book clean applied_data_science_book/
jupyter-book build applied_data_science_book/
```

---

## Chapters

| Chapter | Index file | Status |
|---|---|---|
| Sport Analytics | `sport_analytics.ipynb` | active |
| Fraud Detection | `fraud_detection.md` | active |
| Marketing | `marketing.md` | active |
| Satellite | `satellite.md` | active |
| Algorithmic Trading | `algorithmic_trading.md` | active |
| Risk Assessment | `risk_assesment.md` | active (note: intentional typo in filenames) |
| Quality Control | `quality_control.md` | stub |
| **Information Retrieval, Search and Ranking** | `information_retrieval/index.md` | **new** |

---

## Workflow for Adding Content

1. Drop resource articles / notebooks into the relevant chapter subfolder.
2. Create or update the corresponding `.md` or `.ipynb` section file.
3. Add the new `file:` entry under `sections:` in `_toc.yml`.
4. Build locally to verify: `jupyter-book build applied_data_science_book/`
