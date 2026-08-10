# vcsc

A Value-Compressed Sparse Column/Row (VCSC/VCSR) overlay for [AnnData](https://anndata.readthedocs.io),
implemented in NumPy and accelerated with [Numba](https://numba.pydata.org/).

VCSC/VCSR are compressed-sparse layouts inspired by
[IVSparse's VCSC](https://github.com/Seth-Wolfgang/IVSparse). In addition to the usual
compressed-sparse pointer/index arrays, nonzero values within each major-axis slice
(columns for VCSC, rows for VCSR) are deduplicated: each unique value is stored once,
alongside the list of minor-axis positions that share it. This is a strict memory win
whenever values repeat heavily within a slice — as is typical for integer count
matrices, e.g. single-cell RNA-seq counts.

## Install

```sh
uv sync
```

## Quickstart

```python
import vcsc

adata = ...  # an AnnData object
v = vcsc.from_anndata(adata)     # VCSCArray, from adata.X
v.T                                # transpose -> VCSRArray, free (shared buffers)
v * 2.0                            # scalar multiplication
v.log1p()                          # elementwise log1p
v @ x                              # matrix-vector product
v.to_scipy()                       # decompress back to scipy.sparse.csc_array
```

See `docs/` for full usage and API documentation.

## Development

```sh
uv sync --all-extras --dev
uv run pytest
uv run ruff check .
uv run ty check
uv run sphinx-build -b html docs docs/_build/html
```
