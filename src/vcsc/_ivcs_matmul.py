"""Parallel numba kernels for ``IVCSCArrayNormalized``/``IVCSRArrayNormalized`` @ dense.

Two things make a naive implementation wasteful, and both are addressed here
rather than by materializing anything:

1. **Centering makes the normalized matrix dense.** Every implicit structural
   zero becomes ``-col_mean[j]`` (see :mod:`vcsc._ivcs_norm`), so the
   "normalized matrix" is never actually sparse. But the matmul still is,
   algebraically: for row-scale ``r``, gene-scale ``g``, and centering
   ``c = col_mean``,

       A_norm = broadcast_rows(-c) + Delta,   Delta[i, j] = log10(1 + 1000 * raw[i, j] / r[i] / g[j])

   with ``Delta`` exactly zero off the structural nonzeros (``log10(1) ==
   0``). So ``A_norm @ B = ones(n_rows) (x) (-c @ B) + Delta @ B``, and
   ``B @ A_norm = (B.sum(axis=1)) (x) (-c) + B @ Delta``. The dense
   correction term is ``O(n_cols * k)`` or ``O(n_rows + n_cols)`` -- cheap --
   so only ``Delta @ B`` / ``B @ Delta`` (genuinely ``O(nnz)``) needs a
   kernel at all.

2. **Decoding shouldn't have to happen before multiplying.** ``Delta``'s
   nonzero positions live in ``packed_indices`` (LEB128 delta-varint bytes,
   see :mod:`vcsc._ivcsc`), not a plain ``indices`` array. Each kernel below
   walks that byte stream directly -- decoding a column/row index only long
   enough to immediately multiply-accumulate it into the output row/column,
   never writing a decoded ``indices`` array or a ``Delta`` matrix anywhere.

Parallelization is over the same group-aligned byte chunks
:func:`vcsc._ivcsc._group_chunk_boundaries` computes for decoding: each
chunk owns a contiguous run of groups (and hence of major slices), safe to
process independently. Two shapes of kernel result:

- *major-aligned* (:class:`~vcsc.IVCSRArray` for ``self @ B``,
  :class:`~vcsc.IVCSCArray` for ``B @ self``): the major axis a chunk
  decodes is exactly the axis the output is disjoint over, so each chunk
  writes straight into ``out`` -- except the (at most two) major slices
  that straddle a chunk boundary, which go into a small per-chunk
  boundary buffer (``O(n_chunks * k)``, not ``O(n_rows/cols * k)``) and get
  folded into ``out`` in one short serial pass afterward.
- *scatter* (:class:`~vcsc.IVCSCArray` for ``self @ B``,
  :class:`~vcsc.IVCSRArray` for ``B @ self``): the major axis a chunk
  decodes is *not* the output's disjoint axis, so every nonzero can land in
  any output row/column regardless of chunk. Each chunk instead accumulates
  into its own private full-sized output buffer, summed across chunks at
  the end -- the same thread-local-partial-then-reduce idiom
  :mod:`vcsc._ivcs_norm` already uses for its column-stats scatter passes,
  traded for parallelism at an ``O(n_chunks)`` memory multiplier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numba
import numpy as np

from vcsc._ivcsc import _group_chunk_boundaries, _num_chunks

if TYPE_CHECKING:
    from vcsc._ivcs_norm import _IVCSNormalizedBase

__all__ = ["dense_at_normalized", "normalized_at_dense"]


def _chunking(value_ptr: np.ndarray, packed_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_chunks = _num_chunks(packed_indices.shape[0])
    return _group_chunk_boundaries(value_ptr, packed_indices, n_chunks)


# -- major-aligned: IVCSR, self @ B (out rows == major slices) --------------


@numba.njit(cache=True, parallel=True)
def _ivcsr_matmul_delta(
    major_ptr, values, value_ptr, buf, chunk_group, chunk_byte,
    row_scale, gene_scale, B, out, boundary_vals, boundary_major,
):
    n_chunks = chunk_group.shape[0] - 1
    k = B.shape[1]
    for t in numba.prange(n_chunks):  # ty: ignore[not-iterable]
        g0, g1 = chunk_group[t], chunk_group[t + 1]
        boundary_major[t, 0] = -1
        boundary_major[t, 1] = -1
        if g0 == g1:
            continue
        pos = chunk_byte[t]
        m = int(np.searchsorted(major_ptr, g0, side="right") - 1)
        local = np.zeros(k, dtype=np.float64)
        n_flushed = 0
        for g in range(g0, g1):
            while g >= major_ptr[m + 1]:
                contained = major_ptr[m] >= g0 and major_ptr[m + 1] <= g1
                if contained:
                    out[m, :] += local
                else:
                    boundary_vals[t, n_flushed, :] = local
                    boundary_major[t, n_flushed] = m
                    n_flushed += 1
                local = np.zeros(k, dtype=np.float64)
                m += 1
            v = values[g]
            rs = row_scale[m]
            start, end = value_ptr[g], value_ptr[g + 1]
            prev = np.int64(-1)
            for _kk in range(start, end):
                shift = np.uint64(0)
                result = np.uint64(0)
                while True:
                    b = buf[pos]
                    pos += 1
                    result |= np.uint64(b & 0x7F) << shift
                    if b & 0x80 == 0:
                        break
                    shift += np.uint64(7)
                prev = prev + 1 + np.int64(result)
                col = prev
                gs = gene_scale[col]
                if gs > 0.0:
                    delta = np.log10(1.0 + 1000.0 * (v / rs / gs))
                    for c in range(k):
                        local[c] += delta * B[col, c]
        contained = major_ptr[m] >= g0 and major_ptr[m + 1] <= g1
        if contained:
            out[m, :] += local
        else:
            boundary_vals[t, n_flushed, :] = local
            boundary_major[t, n_flushed] = m


def _matmul_ivcsr(arr, row_scale, gene_scale, B: np.ndarray) -> np.ndarray:
    n_rows = arr.n_major
    k = B.shape[1]
    chunk_group, chunk_byte = _chunking(arr.value_ptr, arr.packed_indices)
    n_chunks = chunk_group.shape[0] - 1
    out = np.zeros((n_rows, k), dtype=np.float64)
    boundary_vals = np.zeros((n_chunks, 2, k), dtype=np.float64)
    boundary_major = np.full((n_chunks, 2), -1, dtype=np.int64)
    _ivcsr_matmul_delta(
        arr.major_ptr, arr.values, arr.value_ptr, arr.packed_indices,
        chunk_group, chunk_byte, row_scale, gene_scale, B, out,
        boundary_vals, boundary_major,
    )
    for t in range(n_chunks):
        for s in range(2):
            m = boundary_major[t, s]
            if m >= 0:
                out[m, :] += boundary_vals[t, s, :]
    return out


# -- major-aligned: IVCSC, B @ self (out cols == major slices) --------------


@numba.njit(cache=True, parallel=True)
def _ivcsc_rmatmul_delta(
    major_ptr, values, value_ptr, buf, chunk_group, chunk_byte,
    row_scale, gene_scale, B, out, boundary_vals, boundary_major,
):
    n_chunks = chunk_group.shape[0] - 1
    p = B.shape[0]
    for t in numba.prange(n_chunks):  # ty: ignore[not-iterable]
        g0, g1 = chunk_group[t], chunk_group[t + 1]
        boundary_major[t, 0] = -1
        boundary_major[t, 1] = -1
        if g0 == g1:
            continue
        pos = chunk_byte[t]
        m = int(np.searchsorted(major_ptr, g0, side="right") - 1)
        local = np.zeros(p, dtype=np.float64)
        n_flushed = 0
        for g in range(g0, g1):
            while g >= major_ptr[m + 1]:
                contained = major_ptr[m] >= g0 and major_ptr[m + 1] <= g1
                if contained:
                    out[:, m] += local
                else:
                    boundary_vals[t, n_flushed, :] = local
                    boundary_major[t, n_flushed] = m
                    n_flushed += 1
                local = np.zeros(p, dtype=np.float64)
                m += 1
            v = values[g]
            gs = gene_scale[m]
            start, end = value_ptr[g], value_ptr[g + 1]
            prev = np.int64(-1)
            for _kk in range(start, end):
                shift = np.uint64(0)
                result = np.uint64(0)
                while True:
                    b = buf[pos]
                    pos += 1
                    result |= np.uint64(b & 0x7F) << shift
                    if b & 0x80 == 0:
                        break
                    shift += np.uint64(7)
                prev = prev + 1 + np.int64(result)
                row = prev
                if gs > 0.0:
                    delta = np.log10(1.0 + 1000.0 * (v / row_scale[row] / gs))
                    for c in range(p):
                        local[c] += delta * B[c, row]
        contained = major_ptr[m] >= g0 and major_ptr[m + 1] <= g1
        if contained:
            out[:, m] += local
        else:
            boundary_vals[t, n_flushed, :] = local
            boundary_major[t, n_flushed] = m


def _rmatmul_ivcsc(arr, row_scale, gene_scale, B: np.ndarray) -> np.ndarray:
    n_cols = arr.n_major
    p = B.shape[0]
    chunk_group, chunk_byte = _chunking(arr.value_ptr, arr.packed_indices)
    n_chunks = chunk_group.shape[0] - 1
    out = np.zeros((p, n_cols), dtype=np.float64)
    boundary_vals = np.zeros((n_chunks, 2, p), dtype=np.float64)
    boundary_major = np.full((n_chunks, 2), -1, dtype=np.int64)
    _ivcsc_rmatmul_delta(
        arr.major_ptr, arr.values, arr.value_ptr, arr.packed_indices,
        chunk_group, chunk_byte, row_scale, gene_scale, B, out,
        boundary_vals, boundary_major,
    )
    for t in range(n_chunks):
        for s in range(2):
            m = boundary_major[t, s]
            if m >= 0:
                out[:, m] += boundary_vals[t, s, :]
    return out


# -- scatter: IVCSC, self @ B (out rows == minor axis, not major) -----------


@numba.njit(cache=True, parallel=True)
def _ivcsc_matmul_delta_scatter(
    major_ptr, values, value_ptr, buf, chunk_group, chunk_byte,
    row_scale, gene_scale, B, partial,
):
    n_chunks = chunk_group.shape[0] - 1
    k = B.shape[1]
    for t in numba.prange(n_chunks):  # ty: ignore[not-iterable]
        g0, g1 = chunk_group[t], chunk_group[t + 1]
        if g0 == g1:
            continue
        out_t = partial[t]
        pos = chunk_byte[t]
        m = int(np.searchsorted(major_ptr, g0, side="right") - 1)
        for g in range(g0, g1):
            while g >= major_ptr[m + 1]:
                m += 1
            v = values[g]
            gs = gene_scale[m]
            start, end = value_ptr[g], value_ptr[g + 1]
            prev = np.int64(-1)
            for _kk in range(start, end):
                shift = np.uint64(0)
                result = np.uint64(0)
                while True:
                    b = buf[pos]
                    pos += 1
                    result |= np.uint64(b & 0x7F) << shift
                    if b & 0x80 == 0:
                        break
                    shift += np.uint64(7)
                prev = prev + 1 + np.int64(result)
                row = prev
                if gs > 0.0:
                    delta = np.log10(1.0 + 1000.0 * (v / row_scale[row] / gs))
                    for c in range(k):
                        out_t[row, c] += delta * B[m, c]


def _matmul_ivcsc(arr, row_scale, gene_scale, B: np.ndarray) -> np.ndarray:
    n_rows = arr.n_minor
    k = B.shape[1]
    chunk_group, chunk_byte = _chunking(arr.value_ptr, arr.packed_indices)
    n_chunks = chunk_group.shape[0] - 1
    partial = np.zeros((max(n_chunks, 1), n_rows, k), dtype=np.float64)
    _ivcsc_matmul_delta_scatter(
        arr.major_ptr, arr.values, arr.value_ptr, arr.packed_indices,
        chunk_group, chunk_byte, row_scale, gene_scale, B, partial,
    )
    return partial.sum(axis=0)


# -- scatter: IVCSR, B @ self (out cols == minor axis, not major) -----------


@numba.njit(cache=True, parallel=True)
def _ivcsr_rmatmul_delta_scatter(
    major_ptr, values, value_ptr, buf, chunk_group, chunk_byte,
    row_scale, gene_scale, B, partial,
):
    n_chunks = chunk_group.shape[0] - 1
    p = B.shape[0]
    for t in numba.prange(n_chunks):  # ty: ignore[not-iterable]
        g0, g1 = chunk_group[t], chunk_group[t + 1]
        if g0 == g1:
            continue
        out_t = partial[t]
        pos = chunk_byte[t]
        m = int(np.searchsorted(major_ptr, g0, side="right") - 1)
        for g in range(g0, g1):
            while g >= major_ptr[m + 1]:
                m += 1
            v = values[g]
            rs = row_scale[m]
            start, end = value_ptr[g], value_ptr[g + 1]
            prev = np.int64(-1)
            for _kk in range(start, end):
                shift = np.uint64(0)
                result = np.uint64(0)
                while True:
                    b = buf[pos]
                    pos += 1
                    result |= np.uint64(b & 0x7F) << shift
                    if b & 0x80 == 0:
                        break
                    shift += np.uint64(7)
                prev = prev + 1 + np.int64(result)
                col = prev
                gs = gene_scale[col]
                if gs > 0.0:
                    delta = np.log10(1.0 + 1000.0 * (v / rs / gs))
                    for c in range(p):
                        out_t[c, col] += delta * B[c, m]


def _rmatmul_ivcsr(arr, row_scale, gene_scale, B: np.ndarray) -> np.ndarray:
    n_cols = arr.n_minor
    p = B.shape[0]
    chunk_group, chunk_byte = _chunking(arr.value_ptr, arr.packed_indices)
    n_chunks = chunk_group.shape[0] - 1
    partial = np.zeros((max(n_chunks, 1), p, n_cols), dtype=np.float64)
    _ivcsr_rmatmul_delta_scatter(
        arr.major_ptr, arr.values, arr.value_ptr, arr.packed_indices,
        chunk_group, chunk_byte, row_scale, gene_scale, B, partial,
    )
    return partial.sum(axis=0)


# -- public entry points: dense correction + sparse delta -------------------


def _prep_dense(other: Any, expect_rows: int) -> tuple[np.ndarray, bool]:
    B = np.asarray(other, dtype=np.float64)
    squeeze = B.ndim == 1
    if squeeze:
        B = B.reshape(-1, 1)
    if B.ndim != 2 or B.shape[0] != expect_rows:
        raise ValueError(f"shape mismatch: expected first dimension {expect_rows}, got {B.shape}")
    return np.ascontiguousarray(B), squeeze


def normalized_at_dense(nview: _IVCSNormalizedBase, other: Any) -> np.ndarray:
    """``nview @ other`` -- normalized-view-on-the-left sparse-dense product."""
    arr = nview._arr
    n_cols = arr.shape[1]
    B, squeeze = _prep_dense(other, n_cols)

    if arr._format == "csr":
        out = _matmul_ivcsr(arr, nview.row_scale, nview.gene_scale, B)
    else:
        out = _matmul_ivcsc(arr, nview.row_scale, nview.gene_scale, B)
    baseline = (-nview.col_mean) @ B  # (k,): every row's implicit-zero contribution
    out += baseline[None, :]
    return out[:, 0] if squeeze else out


def dense_at_normalized(nview: _IVCSNormalizedBase, other: Any) -> np.ndarray:
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
        out = _rmatmul_ivcsr(arr, nview.row_scale, nview.gene_scale, B2)
    else:
        out = _rmatmul_ivcsc(arr, nview.row_scale, nview.gene_scale, B2)
    baseline = B2.sum(axis=1)[:, None] * (-nview.col_mean)[None, :]  # (m, n_cols)
    out += baseline
    return out[0, :] if squeeze else out
