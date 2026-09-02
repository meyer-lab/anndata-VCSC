"""Parallel numba kernels for ``VCSCArrayNormalized``/``VCSRArrayNormalized`` @ dense.

For row-scale ``r``, gene-scale ``g``, and centering ``c = col_mean``,

    A_norm = broadcast_rows(-c) + Delta,   Delta[i, j] = log10(1 + 1000 * raw[i, j] / r[i] / g[j])

with ``Delta`` exactly zero off the structural nonzeros, so
``A_norm @ B = ones(n_rows) (x) (-c @ B) + Delta @ B`` and ``B @ A_norm =
(B.sum(axis=1)) (x) (-c) + B @ Delta``, with only the genuinely ``O(nnz)``
``Delta @ B`` / ``B @ Delta`` needing a kernel.

Both ``self @ B`` and ``B @ self`` need a kernel that's *major-aligned*:
parallelizing safely over the major axis requires the major axis to be
exactly the output's disjoint axis (:class:`~vsparse.VCSRArray` for
``self @ B``, :class:`~vsparse.VCSCArray` for ``B @ self`` -- see
:func:`_vcsr_matmul_delta`/:func:`_vcsc_rmatmul_delta`). The other direction
on a given array (:class:`~vsparse.VCSCArray` for ``self @ B``,
:class:`~vsparse.VCSRArray` for ``B @ self``) doesn't have that alignment --
rather than run a scatter kernel there (thread-local output-shaped
accumulators, reduced across threads: cache-unfriendly, and memory-hungry
enough for wide ``B`` to need throttling), :func:`_get_dual` builds and
caches the *other* VCS format's storage for the same underlying array, once,
via :meth:`~vsparse._base._VCSBase._transpose_major` -- a single global sort
(see :func:`vsparse._construct.transpose_major`), not a per-call cost -- and
every subsequent call in the misaligned direction runs the same
major-aligned kernel against that cached dual instead. ``row_scale``/
``gene_scale``/``col_mean`` are per-row/per-column statistics, so they carry
over unchanged regardless of which physical layout is used to compute
against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numba
import numpy as np

if TYPE_CHECKING:
    from vsparse._vcs_norm import _VCSNormalizedBase

__all__ = ["dense_at_normalized", "normalized_at_dense"]


# -- major-aligned: VCSR, self @ B (out rows == major slices) ---------------


@numba.njit(cache=True, parallel=True)
def _vcsr_matmul_delta(major_ptr, values, value_ptr, indices, row_scale, gene_scale, B, out):
    n_major = major_ptr.shape[0] - 1
    k = B.shape[1]
    for i in numba.prange(n_major):  # ty: ignore[not-iterable]
        rs = row_scale[i]
        for u in range(major_ptr[i], major_ptr[i + 1]):
            v = values[u]
            for kk in range(value_ptr[u], value_ptr[u + 1]):
                col = indices[kk]
                gs = gene_scale[col]
                if gs > 0.0:
                    delta = np.log10(1.0 + 1000.0 * (v / rs / gs))
                    for c in range(k):
                        out[i, c] += delta * B[col, c]


def _matmul_vcsr(arr, row_scale, gene_scale, B: np.ndarray) -> np.ndarray:
    n_rows = arr.n_major
    k = B.shape[1]
    out = np.zeros((n_rows, k), dtype=np.float64)
    _vcsr_matmul_delta(arr.major_ptr, arr.values, arr.value_ptr, arr.indices, row_scale, gene_scale, B, out)
    return out


# -- major-aligned: VCSC, B @ self (out cols == major slices) ---------------
#
# ``B`` arrives as (p, n_rows) and the natural output is (p, n_cols) -- but
# with ``out``/``B`` in that shape, the innermost loop over ``c`` (0..p) hits
# out[c, j] and B[c, row], both strided by the *other* array's full width
# (n_cols/n_rows elements between consecutive c), not the unit stride a
# C-contiguous inner loop needs. Every one of the up to nnz*p inner-loop
# steps is a separate cache line, so this is a per-nonzero cost, not a
# one-off. Transposing both to (n_rows, p)/(n_cols, p) first makes the
# inner loop over ``c`` walk one contiguous row of each -- the transposes
# themselves are O(p * (n_rows + n_cols)), negligible next to the O(nnz * p)
# kernel they speed up.


@numba.njit(cache=True, parallel=True)
def _vcsc_rmatmul_delta(major_ptr, values, value_ptr, indices, row_scale, gene_scale, Bt, out_t):
    n_major = major_ptr.shape[0] - 1
    p = Bt.shape[1]
    for j in numba.prange(n_major):  # ty: ignore[not-iterable]
        gs = gene_scale[j]
        if gs == 0.0:
            continue
        acc = out_t[j]
        for u in range(major_ptr[j], major_ptr[j + 1]):
            v = values[u]
            for kk in range(value_ptr[u], value_ptr[u + 1]):
                row = indices[kk]
                delta = np.log10(1.0 + 1000.0 * (v / row_scale[row] / gs))
                brow = Bt[row]
                for c in range(p):
                    acc[c] += delta * brow[c]


def _rmatmul_vcsc(arr, row_scale, gene_scale, B: np.ndarray) -> np.ndarray:
    n_cols = arr.n_major
    p = B.shape[0]
    Bt = np.ascontiguousarray(B.T)  # (n_rows, p)
    out_t = np.zeros((n_cols, p), dtype=np.float64)
    _vcsc_rmatmul_delta(arr.major_ptr, arr.values, arr.value_ptr, arr.indices, row_scale, gene_scale, Bt, out_t)
    return np.ascontiguousarray(out_t.T)


# -- dual-format cache: gives every call a major-aligned array to run against


def _get_dual(nview: _VCSNormalizedBase):
    """The opposite-format raw array for ``nview``'s wrapped array, built once and cached."""
    if nview._dual_arr is None:
        nview._dual_arr = nview._arr._transpose_major()
    return nview._dual_arr


# -- public entry points: dense correction + sparse delta -------------------


def _prep_dense(other: Any, expect_rows: int) -> tuple[np.ndarray, bool]:
    B = np.asarray(other, dtype=np.float64)
    squeeze = B.ndim == 1
    if squeeze:
        B = B.reshape(-1, 1)
    if B.ndim != 2 or B.shape[0] != expect_rows:
        raise ValueError(f"shape mismatch: expected first dimension {expect_rows}, got {B.shape}")
    return np.ascontiguousarray(B), squeeze


def normalized_at_dense(nview: _VCSNormalizedBase, other: Any) -> np.ndarray:
    """``nview @ other`` -- normalized-view-on-the-left sparse-dense product."""
    arr = nview._arr
    n_cols = arr.shape[1]
    B, squeeze = _prep_dense(other, n_cols)

    # self @ B is major-aligned for VCSR; VCSC needs its VCSR dual.
    src = arr if arr._format == "csr" else _get_dual(nview)
    out = _matmul_vcsr(src, nview.row_scale, nview.gene_scale, B)
    baseline = (-nview.col_mean) @ B  # (k,): every row's implicit-zero contribution
    out += baseline[None, :]
    return out[:, 0] if squeeze else out


def dense_at_normalized(nview: _VCSNormalizedBase, other: Any) -> np.ndarray:
    """``other @ nview`` -- normalized-view-on-the-right dense-sparse product."""
    arr = nview._arr
    n_rows = arr.shape[0]
    other_arr = np.asarray(other, dtype=np.float64)
    squeeze = other_arr.ndim == 1
    B2 = other_arr.reshape(1, -1) if squeeze else other_arr
    if B2.ndim != 2 or B2.shape[1] != n_rows:
        raise ValueError(f"shape mismatch: expected last dimension {n_rows}, got {B2.shape}")
    B2 = np.ascontiguousarray(B2)

    # B @ self is major-aligned for VCSC; VCSR needs its VCSC dual.
    src = arr if arr._format == "csc" else _get_dual(nview)
    out = _rmatmul_vcsc(src, nview.row_scale, nview.gene_scale, B2)
    baseline = B2.sum(axis=1)[:, None] * (-nview.col_mean)[None, :]  # (m, n_cols)
    out += baseline
    return out[0, :] if squeeze else out
