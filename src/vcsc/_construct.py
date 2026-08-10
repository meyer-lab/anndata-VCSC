"""Numba-accelerated construction and decompression of the VCS layout.

The value-compressed sparse (VCS) layout groups the nonzero entries of each
major-axis slice (columns for VCSC, rows for VCSR) by their *value* rather
than storing one value per nonzero. For data with heavy value repetition
(e.g. integer count matrices) this makes the value array much smaller than
the number of nonzeros.

Arrays (major axis of size ``n_major``, ``nnz`` total stored nonzeros,
``n_unique`` total unique (major, value) groups):

``major_ptr``
    ``int64[n_major + 1]``. ``major_ptr[i]:major_ptr[i + 1]`` indexes into
    ``values``/``value_ptr`` for the unique values of major-slice ``i``.
``values``
    ``dtype[n_unique]``. The unique nonzero values, grouped by major slice.
``value_ptr``
    ``int64[n_unique + 1]``. ``value_ptr[k]:value_ptr[k + 1]`` indexes into
    ``indices`` for the minor indices sharing ``values[k]``.
``indices``
    ``int32/int64[nnz]``. Minor-axis indices (rows for VCSC, columns for
    VCSR), grouped by the unique value they belong to.
"""

from __future__ import annotations

import numba
import numpy as np

__all__ = ["compress", "decompress"]


@numba.njit(cache=True)
def _compress(major_ptr_in, minor_indices, data, n_major):
    nnz = minor_indices.shape[0]
    out_indices = np.empty(nnz, dtype=minor_indices.dtype)
    out_values = np.empty(nnz, dtype=data.dtype)
    out_value_ptr = np.empty(nnz + 1, dtype=np.int64)
    out_major_ptr = np.empty(n_major + 1, dtype=np.int64)

    value_count = 0
    pos = 0
    out_major_ptr[0] = 0

    for j in range(n_major):
        start, end = major_ptr_in[j], major_ptr_in[j + 1]
        seg_len = end - start
        if seg_len == 0:
            out_major_ptr[j + 1] = value_count
            continue

        seg_vals = data[start:end]
        seg_minor = minor_indices[start:end]
        order = np.argsort(seg_vals)
        sorted_vals = seg_vals[order]
        sorted_minor = seg_minor[order]

        out_indices[pos : pos + seg_len] = sorted_minor

        out_values[value_count] = sorted_vals[0]
        out_value_ptr[value_count] = pos
        for k in range(1, seg_len):
            if sorted_vals[k] != sorted_vals[k - 1]:
                value_count += 1
                out_values[value_count] = sorted_vals[k]
                out_value_ptr[value_count] = pos + k
        value_count += 1
        pos += seg_len
        out_major_ptr[j + 1] = value_count

    out_value_ptr[value_count] = pos
    return (
        out_major_ptr,
        out_values[:value_count].copy(),
        out_value_ptr[: value_count + 1].copy(),
        out_indices,
    )


@numba.njit(cache=True)
def _decompress(major_ptr, values, value_ptr, indices, n_major, nnz):
    out_indices = np.empty(nnz, dtype=indices.dtype)
    out_data = np.empty(nnz, dtype=values.dtype)
    out_major_ptr = np.empty(n_major + 1, dtype=np.int64)

    pos = 0
    out_major_ptr[0] = 0
    for j in range(n_major):
        u_start, u_end = major_ptr[j], major_ptr[j + 1]
        col_start = pos
        for u in range(u_start, u_end):
            v = values[u]
            i_start, i_end = value_ptr[u], value_ptr[u + 1]
            cnt = i_end - i_start
            out_indices[pos : pos + cnt] = indices[i_start:i_end]
            out_data[pos : pos + cnt] = v
            pos += cnt
        order = np.argsort(out_indices[col_start:pos])
        out_indices[col_start:pos] = out_indices[col_start:pos][order]
        out_data[col_start:pos] = out_data[col_start:pos][order]
        out_major_ptr[j + 1] = pos

    return out_major_ptr, out_indices, out_data


def compress(
    major_ptr: np.ndarray,
    minor_indices: np.ndarray,
    data: np.ndarray,
    n_major: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the VCS layout from a standard compressed-sparse layout.

    Parameters mirror scipy's ``indptr``/``indices``/``data`` for either a
    CSC (major axis = columns) or CSR (major axis = rows) matrix.
    """
    major_ptr = np.ascontiguousarray(major_ptr, dtype=np.int64)
    minor_indices = np.ascontiguousarray(minor_indices)
    data = np.ascontiguousarray(data)
    return _compress(major_ptr, minor_indices, data, n_major)


def decompress(
    major_ptr: np.ndarray,
    values: np.ndarray,
    value_ptr: np.ndarray,
    indices: np.ndarray,
    n_major: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert :func:`compress`, producing standard ``indptr``/``indices``/``data``."""
    nnz = indices.shape[0]
    return _decompress(major_ptr, values, value_ptr, indices, n_major, nnz)
