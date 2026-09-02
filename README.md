# anndata-sc

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

Install from source or via package manager:

```sh
pip install git+https://github.com/meyer-lab/anndata-VCSC.git
# or with uv
uv add git+https://github.com/meyer-lab/anndata-VCSC.git
```

For local development:

```sh
git clone https://github.com/meyer-lab/anndata-VCSC.git
cd anndata-VCSC
uv sync --all-extras --dev
```

## Quickstart

### Working with VCSC / VCSR arrays

```python
import anndata_sc
import scipy.sparse as sp

# Build from an AnnData object or SciPy sparse array
adata = ...  # an AnnData object
v = anndata_sc.from_anndata(adata)                   # VCSCArray (column-compressed) from adata.X
vr = anndata_sc.from_anndata(adata, format="csr")    # VCSRArray (row-compressed)

# Alternatively from scipy sparse matrices
csc = sp.csc_array(adata.X)
v = anndata_sc.VCSCArray.from_scipy(csc)

# Transposition is zero-copy (swaps major/minor axes and shares buffers)
vr = v.T                                       # VCSRArray

# Scalar arithmetic & math
v2 = v * 2.0                                   # Scalar multiplication
v_div = v / 2.0                                # Scalar division
v_neg = -v                                     # Negation
v_log = v.log1p()                              # Elementwise log1p

# Matrix & vector products (Numba-parallelized)
y = v @ x                                      # Matrix-vector: (n_rows, n_cols) @ (n_cols,) -> (n_rows,)
y_left = x @ v                                 # Vector-matrix: (n_rows,) @ (n_rows, n_cols) -> (n_cols,)
Y = v @ B                                      # Matrix-matrix: (n_rows, n_cols) @ (n_cols, k) -> (n_rows, k)
Y_left = B @ v                                 # Matrix-matrix: (k, n_rows) @ (n_rows, n_cols) -> (k, n_cols)

# Slicing
col_slice = v[:, [1, 3, 5]]                    # Fast major-axis slicing (returns VCSCArray)
sub = v[0:10, 0:10]                            # 2D slicing (falls back to scipy)

# Conversion & layers
sp_csc = v.to_scipy()                          # -> scipy.sparse.csc_array (or to_csr())
dense = v.toarray()                            # -> numpy.ndarray
anndata_sc.to_layer(adata, v, key="counts_vcsc")     # Attach to AnnData layer
```

### `VCSCAnnData`: AnnData with direct VCSC/VCSR backing

`anndata_sc.VCSCAnnData` is an `AnnData` subclass whose `X` (and optionally `raw_X`) is backed directly by a `VCSCArray` or `VCSRArray`:

```python
import anndata_sc

va = anndata_sc.VCSCAnnData.from_anndata(adata)      # Compresses X and raw.X
va.X                                           # VCSCArray
va.raw_X                                       # VCSCArray (separate from anndata's .raw)

# Persist to HDF5 (.h5ad) or Zarr with default Blosc2+LZ4 compression
va.write_h5ad("compressed.h5ad")               # Read back with VCSCAnnData.read_h5ad
va2 = anndata_sc.VCSCAnnData.read_h5ad("compressed.h5ad")

va.write_zarr("compressed.zarr")               # Read back with VCSCAnnData.read_zarr
va3 = anndata_sc.VCSCAnnData.read_zarr("compressed.zarr")

# Escape hatch back to standard AnnData
plain = va.to_anndata()
```

### Byte-Packed On-Disk Format (IVCSC / IVCSR)

For smaller files, pass `format="ivcsc"` (or `"ivcsr"`) on write. This byte-packs the minor-axis indices with delta + varint encoding (inspired by [IVSparse's IVCSC](https://github.com/Seth-Wolfgang/IVSparse)).

It is purely an archival storage format: `read_h5ad` / `read_zarr` decompress the indices on load and return a standard `VCSCArray`/`VCSRArray`.

```python
# Write delta+varint byte-packed indices
va.write_h5ad("archived.h5ad", format="ivcsc")

# Reads back directly as an ordinary VCSCAnnData
va_loaded = anndata_sc.VCSCAnnData.read_h5ad("archived.h5ad")
```

### Fast Filtering & Depth Normalization (`load_and_normalize`)

For IVCSR-backed datasets, `anndata_sc.load_and_normalize` bypasses full array decompression for rapid preprocessing, reproducing the filtering and depth normalization from `parafac2.normalize.prepare_dataset`:

- **Cell filtering without decoding**: Computes cell totals in $O(n_{\text{unique}})$ time directly from group sizes without unpacking varint indices.
- **Fused filter & normalization**: Performs gene filtering, cell/gene depth-scaling, and $\log_{10}(1000x + 1)$ transform in parallel passes over the compact CSR representation.

```python
adata_norm = anndata_sc.load_and_normalize(
    "archived.h5ad",
    min_cell_counts=10.0,      # Filter cells with counts <= 10
    gene_threshold=0.05,       # Filter genes with counts <= 0.05 * n_cells
)
```

See `docs/` for full usage guides and API documentation.

## Development

```sh
uv sync --all-extras --dev
uv run pytest
uv run ruff check .
uv run ty check
uv run sphinx-build -b html docs docs/_build/html
```
