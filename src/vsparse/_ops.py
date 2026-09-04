"""Numba-accelerated numerical kernels on the VCS layout."""

from __future__ import annotations

import numba
import numpy as np

__all__ = [
    "major_matmat",
    "major_matvec",
    "minor_matmat",
    "minor_matvec",
    "minor_select_counts",
    "minor_select_fill",
]


@numba.njit(cache=True)
def _major_matvec(major_ptr, values, value_ptr, indices, x, n_major, n_minor):
    """y = A @ x where A's major axis (columns for VCSC) has length n_major.

    x has length n_major, output y has length n_minor.
    """
    y = np.zeros(n_minor, dtype=values.dtype)
    for j in range(n_major):
        xj = x[j]
        if xj == 0:
            continue
        for u in range(major_ptr[j], major_ptr[j + 1]):
            val = values[u] * xj
            for k in range(value_ptr[u], value_ptr[u + 1]):
                y[indices[k]] += val
    return y


@numba.njit(cache=True, parallel=True)
def _minor_matvec(major_ptr, values, value_ptr, indices, x, n_major):
    """y = x @ A, i.e. contract over the minor axis. x has length n_minor.

    Output y has length n_major. Safe to parallelize over major slices since
    each output element is written by exactly one iteration.
    """
    y = np.zeros(n_major, dtype=values.dtype)
    for j in numba.prange(n_major):  # ty: ignore[not-iterable]
        s = 0.0
        for u in range(major_ptr[j], major_ptr[j + 1]):
            acc = 0.0
            for k in range(value_ptr[u], value_ptr[u + 1]):
                acc += x[indices[k]]
            s += values[u] * acc
        y[j] = s
    return y


@numba.njit(cache=True)
def _major_matmat(major_ptr, values, value_ptr, indices, b, n_major, n_minor):
    """Y = A @ B where A's major axis has length n_major, B is (n_major, k)."""
    k = b.shape[1]
    y = np.zeros((n_minor, k), dtype=values.dtype)
    for j in range(n_major):
        for u in range(major_ptr[j], major_ptr[j + 1]):
            val = values[u]
            for idx in range(value_ptr[u], value_ptr[u + 1]):
                row = indices[idx]
                for c in range(k):
                    y[row, c] += val * b[j, c]
    return y


@numba.njit(cache=True, parallel=True)
def _minor_matmat(major_ptr, values, value_ptr, indices, b, n_major):
    """Y = B @ A, i.e. B is (k, n_minor), contracting over the minor axis."""
    k = b.shape[0]
    y = np.zeros((k, n_major), dtype=values.dtype)
    for j in numba.prange(n_major):  # ty: ignore[not-iterable]
        for u in range(major_ptr[j], major_ptr[j + 1]):
            val = values[u]
            for idx in range(value_ptr[u], value_ptr[u + 1]):
                col = indices[idx]
                for c in range(k):
                    y[c, j] += val * b[c, col]
    return y


def major_matvec(major_ptr, values, value_ptr, indices, x, n_major, n_minor):
    return _major_matvec(major_ptr, values, value_ptr, indices, np.asarray(x), n_major, n_minor)


def minor_matvec(major_ptr, values, value_ptr, indices, x, n_major):
    return _minor_matvec(major_ptr, values, value_ptr, indices, np.asarray(x), n_major)


def major_matmat(major_ptr, values, value_ptr, indices, b, n_major, n_minor):
    b = np.ascontiguousarray(b)
    return _major_matmat(major_ptr, values, value_ptr, indices, b, n_major, n_minor)


def minor_matmat(major_ptr, values, value_ptr, indices, b, n_major):
    b = np.ascontiguousarray(b)
    return _minor_matmat(major_ptr, values, value_ptr, indices, b, n_major)


# -- minor-axis selection ----------------------------------------------------


@numba.njit(cache=True, parallel=True)
def _minor_select_counts(value_ptr, indices, fanout, out_counts):
    # A selection may name one index several times, so each stored element
    # contributes ``fanout`` output entries rather than one.
    n_slots = value_ptr.shape[0] - 1
    for u in numba.prange(n_slots):  # ty: ignore[not-iterable]
        c = 0
        for k in range(value_ptr[u], value_ptr[u + 1]):
            c += fanout[indices[k]]
        out_counts[u] = c


@numba.njit(cache=True, parallel=True)
def _minor_select_fill(
    value_ptr, indices, offsets, positions, kept_slots, new_value_ptr, out_indices
):
    for s in numba.prange(kept_slots.shape[0]):  # ty: ignore[not-iterable]
        u = kept_slots[s]
        pos = new_value_ptr[s]  # each slot fills a disjoint range
        for k in range(value_ptr[u], value_ptr[u + 1]):
            o = indices[k]
            for j in range(offsets[o], offsets[o + 1]):  # every destination of o
                out_indices[pos] = positions[j]
                pos += 1


def minor_select_counts(value_ptr, indices, fanout):
    """Output element count per stored (major, value) slot."""
    counts = np.empty(value_ptr.shape[0] - 1, dtype=np.int64)
    _minor_select_counts(value_ptr, indices, fanout, counts)
    return counts


def minor_select_fill(value_ptr, indices, offsets, positions, kept_slots, new_value_ptr, out_indices):
    """Write the remapped minor indices for the surviving slots, in place."""
    _minor_select_fill(
        value_ptr, indices, offsets, positions, kept_slots, new_value_ptr, out_indices
    )
