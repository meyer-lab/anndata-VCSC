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

# Or hold X (and raw.X) directly as a VCSCArray on an AnnData subclass:
va = vcsc.VCSCAnnData.from_anndata(adata)
va.write_h5ad("compressed.h5ad")             # read back with VCSCAnnData.read_h5ad
plain = va.to_anndata()                      # escape hatch back to a normal AnnData
```

`write_h5ad`/`write_zarr` compress every array with Blosc2+LZ4 by default (pass
`dataset_kwargs={}` to disable, or your own `dataset_kwargs` to override).

For smaller files at the cost of extra work on write/read, pass
`format="ivcsc"` to store the IVCSC/IVCSR on-disk format instead -- the same
VCSC/VCSR layout, but with the minor-axis indices delta+varint byte-packed
(inspired by [IVSparse's IVCSC](https://github.com/Seth-Wolfgang/IVSparse)).
It's purely a storage format: `read_h5ad`/`read_zarr` always hand back an
ordinary VCSCArray/VCSRArray, decompressed from IVCSC/IVCSR immediately on
load.

```python
va.write_h5ad("archived.h5ad", format="ivcsc")
vcsc.VCSCAnnData.read_h5ad("archived.h5ad")  # X is a plain VCSCArray again
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
