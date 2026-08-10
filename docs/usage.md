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
