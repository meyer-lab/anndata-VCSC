"""Shared statistics/materialization logic for normalized VCS views.

:mod:`anndata_sc._vcs_norm` (:class:`~vcsc.VCSCArrayNormalized`/:class:`~vcsc.
VCSRArrayNormalized`) wraps a raw VCS array and behaves like the
read-depth-normalized, log-transformed, mean-centered matrix that
:func:`anndata_sc._rapid_load.load_and_normalize` builds -- without ever
materializing it. Centering makes every implicit structural zero a nonzero
value, so a real materialization is an ``n_rows * n_cols`` dense array; the
whole point of a "view" here is to avoid paying for that until (and unless)
the caller actually asks for it.

What's precomputed, once, at construction:

- ``row_scale``: per-row (cell) total raw counts, scaled to a median of 1 --
  the read-depth normalization factor.
- ``gene_scale``: per-column (gene) sum of read-depth-scaled raw counts --
  the per-gene normalization factor used inside the log transform.
- ``col_mean``: the mean, over *all* rows (including implicit zeros, whose
  transformed value is exactly ``log10(1) == 0``), of the transformed value
  in that column -- the centering offset.

Given those three (all ``O(n_rows)``/``O(n_cols)``-sized, not ``O(nnz)``),
any entry's final value is ``log10(1 + 1000 * raw / row_scale / gene_scale)
- col_mean``, computable independently per entry. Computing the statistics
themselves still requires touching every nonzero (twice: once for
``gene_scale``, once more for ``col_mean``, since the transform needs
``gene_scale`` first) -- exactly like :mod:`anndata_sc._rapid_load`'s reference
implementation -- so that part is done with parallel numba kernels below,
specialized per storage format:

- major=columns (VCSC): both passes collapse into one, fully parallel
  over columns with no cross-thread writes -- each column's own elements
  carry everything needed to compute both its ``gene_scale`` and its
  ``col_mean`` (:func:`_column_stats_major_is_col`).
- major=rows (VCSR): each pass is a scatter-add across columns from
  many rows, so it's parallelized like :mod:`anndata_sc._rapid_load`'s
  ``_scaled_col_sums`` -- row-chunked with thread-local partial column
  arrays, reduced by summing across threads (:func:`_scaled_col_sums_vcs`,
  :func:`_transformed_col_sums_vcs`).

These kernels only need ``major_ptr``/``values``/``value_ptr``/``indices``
arrays, from the plain (:mod:`anndata_sc._base`) array types. The matmul kernels
in :mod:`anndata_sc._vcs_matmul` walk that same already-materialized ``indices``
array directly. :class:`VCSCArrayNormalized`/:class:`VCSRArrayNormalized`
each supply their own ``__matmul__``/``__rmatmul__`` wired to those kernels.
"""

from __future__ import annotations

from typing import Any

import numba
import numpy as np

__all__ = ["NormalizedViewBase"]


# -- statistics: major=columns -- fused, no cross-thread writes -------------


@numba.njit(cache=True, parallel=True)
def _column_stats_major_is_col(major_ptr, values, value_ptr, indices, row_scale, n_rows):
    n_major = major_ptr.shape[0] - 1
    gene_scale = np.zeros(n_major, dtype=np.float64)
    col_mean = np.zeros(n_major, dtype=np.float64)
    for j in numba.prange(n_major):  # ty: ignore[not-iterable]
        gs = 0.0
        for u in range(major_ptr[j], major_ptr[j + 1]):
            v = values[u]
            for k in range(value_ptr[u], value_ptr[u + 1]):
                gs += v / row_scale[indices[k]]
        gene_scale[j] = gs
        if gs > 0.0 and n_rows > 0:
            cs = 0.0
            for u in range(major_ptr[j], major_ptr[j + 1]):
                v = values[u]
                for k in range(value_ptr[u], value_ptr[u + 1]):
                    scaled = v / row_scale[indices[k]] / gs
                    cs += np.log10(1.0 + 1000.0 * scaled)
            col_mean[j] = cs / n_rows
    return gene_scale, col_mean


# -- statistics: major=rows -- scatter-add passes ----------------------------


@numba.njit(cache=True, parallel=True)
def _scaled_col_sums_vcs(major_ptr, values, value_ptr, indices, row_scale, n_cols, nthreads):
    n_major = major_ptr.shape[0] - 1
    chunk = (n_major + nthreads - 1) // nthreads
    partial = np.zeros((nthreads, n_cols), dtype=np.float64)
    for t in numba.prange(nthreads):  # ty: ignore[not-iterable]
        start = t * chunk
        end = min(n_major, start + chunk)
        local = partial[t]
        for i in range(start, end):
            rs = row_scale[i]
            for u in range(major_ptr[i], major_ptr[i + 1]):
                v = values[u]
                for k in range(value_ptr[u], value_ptr[u + 1]):
                    local[indices[k]] += v / rs
    return partial.sum(axis=0)


@numba.njit(cache=True, parallel=True)
def _transformed_col_sums_vcs(
    major_ptr, values, value_ptr, indices, row_scale, gene_scale, n_cols, nthreads
):
    n_major = major_ptr.shape[0] - 1
    chunk = (n_major + nthreads - 1) // nthreads
    partial = np.zeros((nthreads, n_cols), dtype=np.float64)
    for t in numba.prange(nthreads):  # ty: ignore[not-iterable]
        start = t * chunk
        end = min(n_major, start + chunk)
        local = partial[t]
        for i in range(start, end):
            rs = row_scale[i]
            for u in range(major_ptr[i], major_ptr[i + 1]):
                v = values[u]
                for k in range(value_ptr[u], value_ptr[u + 1]):
                    c = indices[k]
                    gs = gene_scale[c]
                    if gs > 0.0:
                        scaled = v / rs / gs
                        local[c] += np.log10(1.0 + 1000.0 * scaled)
    return partial.sum(axis=0)


# -- full materialization -----------------------------------------------------
#
# Both kernels start from an ``out`` already filled with ``-col_mean``
# (the value every implicit structural zero takes) and only overwrite the
# entries that are actually stored -- each parallelized over the major axis,
# which owns disjoint rows (major=rows) or columns (major=columns) of
# ``out``, so there's no cross-thread write.


@numba.njit(cache=True, parallel=True)
def _fill_normalized_major_is_col(major_ptr, values, value_ptr, indices, row_scale, gene_scale, col_mean, out):
    n_major = major_ptr.shape[0] - 1
    for j in numba.prange(n_major):  # ty: ignore[not-iterable]
        gs = gene_scale[j]
        if gs == 0.0:
            continue
        cm = col_mean[j]
        for u in range(major_ptr[j], major_ptr[j + 1]):
            v = values[u]
            for k in range(value_ptr[u], value_ptr[u + 1]):
                r = indices[k]
                scaled = v / row_scale[r] / gs
                out[r, j] = np.log10(1.0 + 1000.0 * scaled) - cm


@numba.njit(cache=True, parallel=True)
def _fill_normalized_major_is_row(major_ptr, values, value_ptr, indices, row_scale, gene_scale, col_mean, out):
    n_major = major_ptr.shape[0] - 1
    for i in numba.prange(n_major):  # ty: ignore[not-iterable]
        rs = row_scale[i]
        for u in range(major_ptr[i], major_ptr[i + 1]):
            v = values[u]
            for k in range(value_ptr[u], value_ptr[u + 1]):
                c = indices[k]
                gs = gene_scale[c]
                if gs == 0.0:
                    continue
                scaled = v / rs / gs
                out[i, c] = np.log10(1.0 + 1000.0 * scaled) - col_mean[c]


def _prep_key(key: Any) -> Any:
    """Turn a bare int into a length-1 list, so fancy indexing never drops that axis."""
    if isinstance(key, int | np.integer):
        return [int(key)]
    return key


class NormalizedViewBase:
    """Shared implementation for the normalized VCSC/VCSR views.

    Subclasses fix ``_format`` (``"csc"``/``"csr"``) and supply
    ``__matmul__``/``__rmatmul__`` wired to :mod:`anndata_sc._vcs_matmul`.
    """

    _format: str

    __array_ufunc__ = None

    __slots__ = ("_arr", "col_mean", "gene_scale", "row_scale")

    def __init__(self, arr: Any) -> None:
        if arr._format != self._format:
            raise ValueError(
                f"{type(self).__name__} wraps a {self._format!r}-format array, "
                f"got {type(arr).__name__}"
            )
        self._arr = arr

        n_rows, n_cols = arr.shape
        row_totals = np.asarray(arr.sum(axis=1), dtype=np.float64)
        median = float(np.median(row_totals)) if row_totals.shape[0] else 0.0
        if median > 0.0:
            row_scale = row_totals / median
            row_scale[row_scale == 0.0] = 1.0
        else:
            row_scale = np.ones(n_rows, dtype=np.float64)
        self.row_scale = row_scale

        indices = arr.indices  # decode once; shared by both statistics passes below
        if self._format == "csc":
            gene_scale, col_mean = _column_stats_major_is_col(
                arr.major_ptr, arr.values, arr.value_ptr, indices, row_scale, n_rows
            )
        else:
            nthreads = numba.get_num_threads()
            gene_scale = _scaled_col_sums_vcs(
                arr.major_ptr, arr.values, arr.value_ptr, indices, row_scale, n_cols, nthreads
            )
            col_sum = _transformed_col_sums_vcs(
                arr.major_ptr, arr.values, arr.value_ptr, indices, row_scale, gene_scale, n_cols, nthreads
            )
            col_mean = col_sum / n_rows if n_rows > 0 else np.zeros(n_cols, dtype=np.float64)
        self.gene_scale = gene_scale
        self.col_mean = col_mean

    @property
    def shape(self) -> tuple[int, int]:
        return self._arr.shape

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float64)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} shape={self.shape} dtype={self.dtype}>"

    # -- materialization ------------------------------------------------------

    def toarray(self) -> np.ndarray:
        """The full normalized, log-transformed, mean-centered matrix, densely."""
        n_rows, n_cols = self.shape
        out = np.broadcast_to(-self.col_mean, (n_rows, n_cols)).copy()
        arr = self._arr
        if self._format == "csc":
            _fill_normalized_major_is_col(
                arr.major_ptr, arr.values, arr.value_ptr, arr.indices,
                self.row_scale, self.gene_scale, self.col_mean, out,
            )
        else:
            _fill_normalized_major_is_row(
                arr.major_ptr, arr.values, arr.value_ptr, arr.indices,
                self.row_scale, self.gene_scale, self.col_mean, out,
            )
        return out

    # -- on-the-fly elementwise access ------------------------------------------

    def __getitem__(self, key: Any) -> np.ndarray:
        """Compute just the requested sub-block, on the fly, from the raw data.

        Only the raw counts for the requested rows/columns are ever
        decompressed (via the underlying array's own indexing, which stays
        compact for a major-axis-only slice); the transform/centering
        formula is then applied to that small block directly, using the
        precomputed per-row/per-column statistics -- never the full matrix.
        """
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError(f"{type(self).__name__} arrays are 2-D")
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)

        row_key = _prep_key(row_key)
        col_key = _prep_key(col_key)

        dense_raw = self._arr[row_key, col_key].toarray().astype(np.float64)
        rs = np.asarray(self.row_scale)[row_key].reshape(-1, 1)
        gs = np.asarray(self.gene_scale)[col_key].reshape(1, -1)
        cm = np.asarray(self.col_mean)[col_key].reshape(1, -1)

        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = np.where(gs > 0.0, dense_raw / rs / gs, 0.0)
        return np.log10(1.0 + 1000.0 * scaled) - cm

    # -- explicitly-unsupported operations ------------------------------------

    def _unsupported(self, op: str) -> Any:
        raise RuntimeError(
            f"{op} is not supported on {type(self).__name__}. "
            "Call .toarray() first if you need it."
        )

    def __add__(self, other: Any) -> Any:
        return self._unsupported("addition")

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:
        return self._unsupported("subtraction")

    def __mul__(self, other: Any) -> Any:
        return self._unsupported("multiplication")

    __rmul__ = __mul__
