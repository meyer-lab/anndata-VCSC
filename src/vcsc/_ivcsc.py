"""Byte-packed delta encoding of VCS indices, for compact on-disk storage.

This backs the "IVCSC"/"IVCSR" file format (named after `IVSparse's IVCSC
<https://github.com/Seth-Wolfgang/IVSparse>`_, which inspired it): the same
VCSC/VCSR layout, but with the ``indices`` array replaced by a variable-
length-integer (LEB128), delta-encoded byte stream, which is typically much
smaller on disk than a plain int32/int64 array.

Indices within each (major-slice, value) group are unordered with respect to
the original layout -- decompression re-sorts them per major slice regardless
(see :mod:`vcsc._construct`) -- so packing is free to sort each group
ascending before delta-encoding it. That keeps every delta non-negative and
usually small, which is what makes the byte packing worth doing.

This module only implements the byte codec; :mod:`vcsc._io` uses it to decode
a packed group straight back into a plain VCSCArray/VCSRArray with an
ordinary ``indices`` array on read, so none of the compute paths in
:mod:`vcsc._ops` need to know this encoding exists.
"""

from __future__ import annotations

import numba
import numpy as np

__all__ = ["pack_indices", "unpack_indices"]


@numba.njit(cache=True)
def _pack(value_ptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    n_groups = value_ptr.shape[0] - 1
    nnz = indices.shape[0]
    # Worst case (index deltas needing the full 10-byte varint width for a
    # 64-bit value) plus one byte per empty group.
    buf = np.empty(nnz * 10 + n_groups, dtype=np.uint8)
    pos = 0
    for g in range(n_groups):
        start, end = value_ptr[g], value_ptr[g + 1]
        seg = np.sort(indices[start:end])
        prev = np.int64(-1)
        for k in range(seg.shape[0]):
            cur = np.int64(seg[k])
            v = np.uint64(cur - prev - 1)
            prev = cur
            while v >= 0x80:
                buf[pos] = np.uint8((v & 0x7F) | 0x80)
                pos += 1
                v >>= np.uint64(7)
            buf[pos] = np.uint8(v)
            pos += 1
    return buf[:pos].copy()


@numba.njit(cache=True)
def _unpack(value_ptr: np.ndarray, buf: np.ndarray, out: np.ndarray) -> None:
    n_groups = value_ptr.shape[0] - 1
    pos = 0
    for g in range(n_groups):
        start, end = value_ptr[g], value_ptr[g + 1]
        prev = np.int64(-1)
        for k in range(start, end):
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
            out[k] = prev


def pack_indices(value_ptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Delta+varint encode ``indices``, grouped by ``value_ptr``, into ``uint8`` bytes."""
    value_ptr = np.ascontiguousarray(value_ptr, dtype=np.int64)
    indices = np.ascontiguousarray(indices)
    return _pack(value_ptr, indices)


def unpack_indices(value_ptr: np.ndarray, packed: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Invert :func:`pack_indices`, producing an ``indices`` array of the given dtype."""
    value_ptr = np.ascontiguousarray(value_ptr, dtype=np.int64)
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    out = np.empty(int(value_ptr[-1]), dtype=dtype)
    _unpack(value_ptr, packed, out)
    return out
