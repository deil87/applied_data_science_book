# Repo for the Applied DS Book

This is currently a work in progress, but feel free to take a look: 
https://deil87.github.io/applied_data_science_book/intro.html

## Local installation

```
source adsb_env/bin/activate  
pip install -r requirements.txt 
```

In order to build the book run this in the root directory:

jupyter-book build applied_data_science_book/.

(known issue, LTR folder is automatilly being picked up and we receive warnings)

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