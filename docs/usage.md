# Usage

## Converting an AnnData object

```python
import anndata as ad
import vcsc

adata = ad.read_h5ad("data.h5ad")

# From adata.X, as a VCSCArray (column-compressed)
v = vcsc.from_anndata(adata)

# From adata.raw.X, as a VCSRArray (row-compressed)
v = vcsc.from_anndata(adata, use_raw=True, format="csr")

# From a specific layer
v = vcsc.from_anndata(adata, layer="counts")
```

## Converting from/to scipy sparse

```python
import scipy.sparse as sp
from vcsc import VCSCArray

csc = sp.csc_array(dense_or_sparse_matrix)
v = VCSCArray.from_scipy(csc)

back = v.to_scipy()   # -> scipy.sparse.csc_array
dense = v.toarray()   # -> numpy.ndarray
```

## Supported operations

```python
v.T                 # transpose (free: returns a VCSRArray sharing buffers)
v * 2.0              # scalar multiplication
v / 2.0              # scalar division
v.log1p()            # elementwise log1p
v @ x                # matrix-vector product, x: 1-D array of length n_cols
x @ v                # vector-matrix product, x: 1-D array of length n_rows
v @ B                # matrix-matrix product, B: 2-D array with B.shape[0] == n_cols
B @ v                # matrix-matrix product, B: 2-D array with B.shape[1] == n_rows
v[:, [1, 3, 5]]       # major-axis (column, for VCSCArray) indexing
v[0:2, 0:2]           # general 2-D indexing (falls back to scipy internally)
```

Attach a decompressed result back onto an `AnnData` object as a layer:

```python
vcsc.to_layer(adata, v, key="vcsc_roundtrip")
```

## `VCSCAnnData`: X backed directly by a VCSCArray

`vcsc.VCSCAnnData` is an `AnnData` subclass whose `X` (and, separately, `raw_X`) *is* a
`VCSCArray`/`VCSRArray`, not a scipy array:

```python
import vcsc

va = vcsc.VCSCAnnData.from_anndata(adata)  # compresses X and raw.X
va.X       # a VCSCArray
va.raw_X   # a VCSCArray (kept separately from anndata's own `.raw`)

# Persist -- read back with the matching classmethod, not anndata.read_h5ad/read_zarr
va.write_h5ad("compressed.h5ad")
va2 = vcsc.VCSCAnnData.read_h5ad("compressed.h5ad")

va.write_zarr("compressed.zarr")
va3 = vcsc.VCSCAnnData.read_zarr("compressed.zarr")

# Escape hatch: decompress to a normal, fully-featured AnnData
plain = va.to_anndata()
```

Standard `AnnData` validates every array assigned to `X`/`layers`/etc. against a fixed
allowlist of types, so a plain `AnnData` cannot hold a `VCSCArray` in `X`.
`VCSCAnnData` works around this by overriding the `X` property; as a consequence,
operations that need anndata's normal per-element type dispatch on `X` -- slicing into
views, concatenation, most of the scanpy/anndata ecosystem -- are **not** supported while
`X` is VCSC/VCSR-backed. Call `.to_anndata()` first if you need those.

`VCSCArray`/`VCSRArray` are also registered with anndata's on-disk IO registry
directly, so they round-trip through `write_elem`/`read_elem` even when nested inside
a plain `AnnData`'s `.uns`, independent of `VCSCAnnData`.
