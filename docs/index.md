# anndata-sc

A Value-Compressed Sparse Column/Row overlay for [AnnData](https://anndata.readthedocs.io),
implemented in NumPy and accelerated with [Numba](https://numba.pydata.org/).

VCSC/VCSR are compressed-sparse layouts inspired by
[IVSparse's VCSC](https://github.com/Seth-Wolfgang/IVSparse). In addition to the usual
compressed-sparse pointer/index arrays, nonzero values within each major-axis slice
(columns for VCSC, rows for VCSR) are deduplicated: each unique value is stored once,
alongside the list of minor-axis positions that share it. This is a strict memory win
whenever values repeat heavily within a slice — as is typical for integer count
matrices, e.g. single-cell RNA-seq counts.

```{toctree}
:maxdepth: 2

usage
api
```
