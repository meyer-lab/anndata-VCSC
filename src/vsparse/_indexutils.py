"""Tiny indexing helpers shared by the VCS and IVCS array types."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["is_full_slice", "normalize_major_idx", "smallest_index_dtype"]

_INT32_MAX = np.iinfo(np.int32).max


def smallest_index_dtype(n: int) -> np.dtype:
    """Narrowest signed integer dtype that can address an axis of length ``n``.

    Index arrays are sized by the axis they point *into*, not by the array
    they live next to: minor-axis ``indices`` are bounded by ``n_minor``
    (typically a gene count, comfortably int32) even when the array holds
    more than ``INT32_MAX`` nonzeros. Keying each index array off its own
    bound is what keeps a large-nnz array from paying int64 for indices that
    never need it.
    """
    return np.dtype(np.int32) if n <= _INT32_MAX else np.dtype(np.int64)


def is_full_slice(key: Any) -> bool:
    return isinstance(key, slice) and key.start is None and key.stop is None and key.step is None


def normalize_major_idx(key: Any, n_major: int) -> np.ndarray:
    """Turn a major-axis index (slice/int/bool mask/int array) into a plain int array."""
    if isinstance(key, slice):
        return np.arange(*key.indices(n_major))
    if isinstance(key, int | np.integer):
        idx = int(key)
        if idx < 0:
            idx += n_major
        if not (0 <= idx < n_major):
            raise IndexError(f"index {key} out of bounds for axis of size {n_major}")
        return np.array([idx])
    arr = np.asarray(key)
    if arr.dtype == bool:
        if arr.shape[0] != n_major:
            raise IndexError("boolean index does not match major axis length")
        return np.nonzero(arr)[0]
    arr = arr.astype(np.int64)
    arr = np.where(arr < 0, arr + n_major, arr)
    if arr.size and ((arr < 0).any() or (arr >= n_major).any()):
        raise IndexError("index out of bounds for major axis")
    return arr
