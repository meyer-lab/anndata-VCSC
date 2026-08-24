"""Parallel numba kernels for ``VCSCArrayNormalized``/``VCSRArrayNormalized`` @ dense.

Same algebraic decomposition as :mod:`vcsc._ivcs_matmul` (see its module
docstring for the derivation): for row-scale ``r``, gene-scale ``g``, and
centering ``c = col_mean``,

    A_norm = broadcast_rows(-c) + Delta,   Delta[i, j] = log10(1 + 1000 * raw[i, j] / r[i] / g[j])

with ``Delta`` exactly zero off the structural nonzeros, so
``A_norm @ B = ones(n_rows) (x) (-c @ B) + Delta @ B`` and ``B @ A_norm =
(B.sum(axis=1)) (x) (-c) + B @ Delta``, with only the genuinely ``O(nnz)``
``Delta @ B`` / ``B @ Delta`` needing a kernel.

Unlike IVCSC/IVCSR, :class:`~vcsc.VCSCArray`/:class:`~vcsc.VCSRArray` keep
``indices`` as a plain, already-decoded int array -- there is no byte stream
to walk, so these kernels are a direct per-nonzero loop (no varint decode,
no group-aligned chunk bookkeeping). Two shapes of kernel result, exactly as
in :mod:`vcsc._ivcs_matmul`:

- *major-aligned* (:class:`~vcsc.VCSRArray` for ``self @ B``,
  :class:`~vcsc.VCSCArray` for ``B @ self``): the major axis is exactly the
  output's disjoint axis, so parallelizing over major slices
  (``numba.prange``) is directly safe -- no boundary bookkeeping needed at
  all, since (unlike a byte stream) each major slice's extent is already
  known without decoding anything before it.
- *scatter* (:class:`~vcsc.VCSCArray` for ``self @ B``,
  :class:`~vcsc.VCSRArray` for ``B @ self``): the major axis a chunk visits
  is not the output's disjoint axis, so each chunk accumulates into its own
  private output-shaped buffer, summed across chunks at the end -- the same
  thread-local-partial-then-reduce idiom used throughout this package. Chunk
  boundaries are chosen along the major axis to balance *nonzero count* per
  chunk (:func:`_nnz_balanced_major_chunks`), since major slices can vary
  widely in size. Peak memory is bounded by the same fixed budget as
  :mod:`vcsc._ivcs_matmul` (:func:`vcsc._ivcs_matmul._scatter_layout`,
  reused here), shrinking ``n_chunks`` and/or column-blocking ``B``/``Bl``
  rather than ever scaling accumulator memory with ``k``/``p``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numba
import numpy as np

from vcsc._ivcs_matmul import _scatter_layout

if TYPE_CHECKING:
    from vcsc._norm_common import NormalizedViewBase

__all__ = ["dense_at_normalized", "normalized_at_dense"]


def _nnz_balanced_major_chunks(major_ptr: np.ndarray, value_ptr: np.ndarray, n_chunks: int) -> np.ndarray:
    """Major-axis boundaries splitting ``[0, n_major]`` into ``n_chunks`` runs of ~equal nnz."""
    n_major = major_ptr.shape[0] - 1
    n_chunks = max(1, min(n_chunks, max(n_major, 1)))
    if n_major == 0 or n_chunks == 1:
        return np.array([0, n_major], dtype=np.int64)
    cum = value_ptr[major_ptr]  # cumulative nnz at each major boundary (length n_major + 1)
    total = cum[-1]
    if total == 0:
        return np.array([0, n_major], dtype=np.int64)
    targets = np.linspace(0, total, n_chunks + 1)[1:-1]
    bounds = np.searchsorted(cum, targets, side="left").astype(np.int64)
    return np.concatenate(([0], bounds, [n_major]))


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


@numba.njit(cache=True, parallel=True)
def _vcsc_rmatmul_delta(major_ptr, values, value_ptr, indices, row_scale, gene_scale, B, out):
    n_major = major_ptr.shape[0] - 1
    p = B.shape[0]
    for j in numba.prange(n_major):  # ty: ignore[not-iterable]
        gs = gene_scale[j]
        if gs == 0.0:
            continue
        for u in range(major_ptr[j], major_ptr[j + 1]):
            v = values[u]
            for kk in range(value_ptr[u], value_ptr[u + 1]):
                row = indices[kk]
                delta = np.log10(1.0 + 1000.0 * (v / row_scale[row] / gs))
                for c in range(p):
                    out[c, j] += delta * B[c, row]


def _rmatmul_vcsc(arr, row_scale, gene_scale, B: np.ndarray) -> np.ndarray:
    n_cols = arr.n_major
    p = B.shape[0]
    out = np.zeros((p, n_cols), dtype=np.float64)
    _vcsc_rmatmul_delta(arr.major_ptr, arr.values, arr.value_ptr, arr.indices, row_scale, gene_scale, B, out)
    return out


# -- scatter: VCSC, self @ B (out rows == minor axis, not major) ------------


@numba.njit(cache=True, parallel=True)
def _vcsc_matmul_delta_scatter(major_ptr, values, value_ptr, indices, row_scale, gene_scale, B, partial, chunk_major):
    n_chunks = chunk_major.shape[0] - 1
    k = B.shape[1]
    for t in numba.prange(n_chunks):  # ty: ignore[not-iterable]
        j0, j1 = chunk_major[t], chunk_major[t + 1]
        if j0 == j1:
            continue
        out_t = partial[t]
        for j in range(j0, j1):
            gs = gene_scale[j]
            if gs == 0.0:
                continue
            for u in range(major_ptr[j], major_ptr[j + 1]):
                v = values[u]
                for kk in range(value_ptr[u], value_ptr[u + 1]):
                    row = indices[kk]
                    delta = np.log10(1.0 + 1000.0 * (v / row_scale[row] / gs))
                    for c in range(k):
                        out_t[row, c] += delta * B[j, c]


def _matmul_vcsc(arr, row_scale, gene_scale, B: np.ndarray) -> np.ndarray:
    n_rows = arr.n_minor
    k = B.shape[1]
    n_chunks_ideal = numba.get_num_threads()
    n_chunks, k_block = _scatter_layout(n_chunks_ideal, n_rows, k)
    chunk_major = _nnz_balanced_major_chunks(arr.major_ptr, arr.value_ptr, n_chunks)
    n_chunks = chunk_major.shape[0] - 1

    out = np.empty((n_rows, k), dtype=np.float64)
    partial = np.zeros((n_chunks, n_rows, k_block), dtype=np.float64)  # freshly zeroed (lazy pages)
    for i, start in enumerate(range(0, k, k_block)):
        end = min(k, start + k_block)
        width = end - start
        block = partial[:, :, :width]
        if i > 0:  # reused buffer: previous block's contents need clearing
            block[:] = 0.0
        _vcsc_matmul_delta_scatter(
            arr.major_ptr, arr.values, arr.value_ptr, arr.indices,
            row_scale, gene_scale, np.ascontiguousarray(B[:, start:end]), block, chunk_major,
        )
        out[:, start:end] = block.sum(axis=0)
    return out


# -- scatter: VCSR, B @ self (out cols == minor axis, not major) ------------


@numba.njit(cache=True, parallel=True)
def _vcsr_rmatmul_delta_scatter(major_ptr, values, value_ptr, indices, row_scale, gene_scale, B, partial, chunk_major):
    n_chunks = chunk_major.shape[0] - 1
    p = B.shape[0]
    for t in numba.prange(n_chunks):  # ty: ignore[not-iterable]
        i0, i1 = chunk_major[t], chunk_major[t + 1]
        if i0 == i1:
            continue
        out_t = partial[t]
        for i in range(i0, i1):
            rs = row_scale[i]
            for u in range(major_ptr[i], major_ptr[i + 1]):
                v = values[u]
                for kk in range(value_ptr[u], value_ptr[u + 1]):
                    col = indices[kk]
                    gs = gene_scale[col]
                    if gs > 0.0:
                        delta = np.log10(1.0 + 1000.0 * (v / rs / gs))
                        for c in range(p):
                            out_t[c, col] += delta * B[c, i]


def _rmatmul_vcsr(arr, row_scale, gene_scale, B: np.ndarray) -> np.ndarray:
    n_cols = arr.n_minor
    p = B.shape[0]
    n_chunks_ideal = numba.get_num_threads()
    n_chunks, p_block = _scatter_layout(n_chunks_ideal, n_cols, p)
    chunk_major = _nnz_balanced_major_chunks(arr.major_ptr, arr.value_ptr, n_chunks)
    n_chunks = chunk_major.shape[0] - 1

    out = np.empty((p, n_cols), dtype=np.float64)
    partial = np.zeros((n_chunks, p_block, n_cols), dtype=np.float64)  # freshly zeroed (lazy pages)
    for i, start in enumerate(range(0, p, p_block)):
        end = min(p, start + p_block)
        width = end - start
        block = partial[:, :width, :]
        if i > 0:  # reused buffer: previous block's contents need clearing
            block[:] = 0.0
        _vcsr_rmatmul_delta_scatter(
            arr.major_ptr, arr.values, arr.value_ptr, arr.indices,
            row_scale, gene_scale, np.ascontiguousarray(B[start:end, :]), block, chunk_major,
        )
        out[start:end, :] = block.sum(axis=0)
    return out


# -- public entry points: dense correction + sparse delta -------------------


def _prep_dense(other: Any, expect_rows: int) -> tuple[np.ndarray, bool]:
    B = np.asarray(other, dtype=np.float64)
    squeeze = B.ndim == 1
    if squeeze:
        B = B.reshape(-1, 1)
    if B.ndim != 2 or B.shape[0] != expect_rows:
        raise ValueError(f"shape mismatch: expected first dimension {expect_rows}, got {B.shape}")
    return np.ascontiguousarray(B), squeeze


def normalized_at_dense(nview: NormalizedViewBase, other: Any) -> np.ndarray:
    """``nview @ other`` -- normalized-view-on-the-left sparse-dense product."""
    arr = nview._arr
    n_cols = arr.shape[1]
    B, squeeze = _prep_dense(other, n_cols)

    if arr._format == "csr":
        out = _matmul_vcsr(arr, nview.row_scale, nview.gene_scale, B)
    else:
        out = _matmul_vcsc(arr, nview.row_scale, nview.gene_scale, B)
    baseline = (-nview.col_mean) @ B  # (k,): every row's implicit-zero contribution
    out += baseline[None, :]
    return out[:, 0] if squeeze else out


def dense_at_normalized(nview: NormalizedViewBase, other: Any) -> np.ndarray:
    """``other @ nview`` -- normalized-view-on-the-right dense-sparse product."""
    arr = nview._arr
    n_rows = arr.shape[0]
    other_arr = np.asarray(other, dtype=np.float64)
    squeeze = other_arr.ndim == 1
    B2 = other_arr.reshape(1, -1) if squeeze else other_arr
    if B2.ndim != 2 or B2.shape[1] != n_rows:
        raise ValueError(f"shape mismatch: expected last dimension {n_rows}, got {B2.shape}")
    B2 = np.ascontiguousarray(B2)

    if arr._format == "csr":
        out = _rmatmul_vcsr(arr, nview.row_scale, nview.gene_scale, B2)
    else:
        out = _rmatmul_vcsc(arr, nview.row_scale, nview.gene_scale, B2)
    baseline = B2.sum(axis=1)[:, None] * (-nview.col_mean)[None, :]  # (m, n_cols)
    out += baseline
    return out[0, :] if squeeze else out
