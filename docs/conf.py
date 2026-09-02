from __future__ import annotations

import vsparse

project = "vsparse"
copyright = "2026, Meyer lab"
author = "Meyer lab"
release = vsparse.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

myst_enable_extensions = ["dollarmath", "amsmath"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
}

html_theme = "furo"
html_static_path = []
