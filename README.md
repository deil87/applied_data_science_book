# Repo for the Applied DS Book

This is currently a work in progress, but feel free to take a look: 
https://deil87.github.io/applied_data_science_book/intro.html

## Build and run locally

A convenience script handles everything in one step:

```bash
./run_book.sh            # incremental build (uses cached notebook outputs)
./run_book.sh --clean    # clean build (clears all cached outputs first)
```

The script will:
1. Create the virtual environment (`adsb_env/`) if it does not exist.
2. Install / sync all dependencies from `applied_data_science_book/requirements.txt`.
3. Build the book with `jupyter-book`.
4. Open the result in your default browser automatically.

### Manual steps (if you prefer)

### 1. Prerequisites

- Python 3.10+ installed on your machine.
- All commands below are run from the **repository root** (`applied_ds_book/`).

### 2. Create and activate the virtual environment

```bash
python -m venv adsb_env
source adsb_env/bin/activate        # macOS / Linux
# adsb_env\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r applied_data_science_book/requirements.txt
```

### 4. Build the book

```bash
jupyter-book build applied_data_science_book/
```

For a **clean rebuild** (clears all cached notebook outputs):

```bash
jupyter-book clean applied_data_science_book/
jupyter-book build applied_data_science_book/
```

### 5. Open in a browser

```bash
open applied_data_science_book/_build/html/index.html   # macOS
# xdg-open applied_data_science_book/_build/html/index.html  # Linux
# start applied_data_science_book/_build/html/index.html     # Windows
```

Or paste the path directly into your browser:

```
file:///path/to/applied_ds_book/applied_data_science_book/_build/html/index.html
```

> **Known issue:** The legacy `Learning to rank/` folder (space in name) is automatically
> picked up by Jupyter Book and generates build warnings. It can safely be ignored.

## Remote deployment to Github Pages

Deployment is done through Github Actions. See /.github/workflows/static.yml

## Getting started

### Adding a chapter or section:

See _toc.yml

Jupyter-book will find corresponding .ipynb or .md files and render them as part of the book


## Dependency hell

 jupyter-book might have many python dependencies as for each page/jupyter notebook we might want to have different dependencies, is it correct?

Yes, your understanding is correct: a **Jupyter Book might require many Python dependencies** because each page or Jupyter notebook could potentially need different libraries or versions.

### Why is this the case?

- Each notebook in a Jupyter Book can contain code that **depends on different Python packages** depending on the analysis or subject.
- Unlike a single script or project, you may have notebooks covering diverse topics, like data visualization, machine learning, or bioinformatics, each requiring specific libraries.
- This can lead to a larger, more complex set of dependencies overall to ensure every notebook runs properly.


### Common approaches to manage this:

- **Single consolidated environment:** You collect all dependencies needed across all notebooks in one big environment (e.g., one `requirements.txt` or Conda env) so that any notebook can run without missing packages.
- **Multiple environments:** You may maintain separate environments for different parts of the book if notebooks are very different (more complex to maintain).
- **Use of Binder or Docker:** For reproducibility, Jupyter Book supports integration with tools like Binder or Docker where you specify dependencies in an environment file; this installs everything upfront.


### Practical advice:

- It's best to **list all required packages explicitly** in a main environment file.
- For maintainability and reproducibility, **pin exact versions** of dependencies.
- Consider tools like `pip-tools` or Conda environment files to simplify managing complex dependencies across notebooks.

***

### Summary:

- Jupyter Book **can have many dependencies collectively**, as each notebook may require different Python packages.
- Often all are installed in one environment for simplicity.

### Backlog

Things to incorporate into the book:
- see `backlog` folder