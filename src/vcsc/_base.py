"""Value-Compressed Sparse Column/Row array types.

VCSC/VCSR are compressed-sparse layouts, inspired by IVSparse's VCSC
(https://github.com/Seth-Wolfgang/IVSparse), that additionally deduplicate
the stored values within each major-axis slice (columns for VCSC, rows for
VCSR). Instead of one stored value per nonzero, each unique value in a
slice is stored once alongside the list of minor-axis indices that share
it. This is a strict win in memory whenever values repeat heavily within a
slice, as is common for integer count matrices (e.g. single-cell RNA-seq
counts).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from vcsc import _construct, _ops
from vcsc._indexutils import is_full_slice as _is_full_slice
from vcsc._indexutils import normalize_major_idx as _normalize_major_idx

__all__ = ["VCSCArray", "VCSRArray"]


class _VCSBase:
    """Shared implementation for :class:`VCSCArray` and :class:`VCSRArray`.

    Subclasses fix ``_format`` to ``"csc"`` or ``"csr"``, which determines
    whether the major axis (the axis whose slices are value-compressed) is
    columns or rows.
    """

    _format: str

    # Tell numpy to defer binary operators (e.g. ndarray @ VCSCArray) to our
    # __rmatmul__/__rmul__ instead of trying to broadcast us as an ndarray.
    __array_ufunc__ = None

    __slots__ = ("indices", "major_ptr", "shape", "value_ptr", "values")

    def __init__(
        self,
        shape: tuple[int, int],
        major_ptr: np.ndarray,
        values: np.ndarray,
        value_ptr: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        n_major, n_minor = self._swap(shape)
        major_ptr = np.asarray(major_ptr, dtype=np.int64)
        value_ptr = np.asarray(value_ptr, dtype=np.int64)
        values = np.asarray(values)
        indices = np.asarray(indices)

        if major_ptr.shape[0] != n_major + 1:
            raise ValueError(
                f"major_ptr has length {major_ptr.shape[0]}, expected {n_major + 1}"
            )
        if value_ptr.shape[0] != values.shape[0] + 1:
            raise ValueError("value_ptr must have length len(values) + 1")
        if major_ptr[-1] != values.shape[0]:
            raise ValueError("major_ptr[-1] must equal len(values)")
        if value_ptr[-1] != indices.shape[0]:
            raise ValueError("value_ptr[-1] must equal len(indices)")
        if indices.shape[0] and (indices.min() < 0 or indices.max() >= n_minor):
            raise ValueError("indices out of bounds for the given shape")

        self.shape = (int(shape[0]), int(shape[1]))
        self.major_ptr = major_ptr
        self.values = values
        self.value_ptr = value_ptr
        self.indices = indices

    # -- axis bookkeeping ------------------------------------------------

    @classmethod
    def _swap(cls, shape: tuple[int, int]) -> tuple[int, int]:
        """Return ``(n_major, n_minor)`` for the given ``(n_rows, n_cols)``."""
        n_rows, n_cols = shape
        return (n_cols, n_rows) if cls._format == "csc" else (n_rows, n_cols)

    @property
    def n_major(self) -> int:
        return self._swap(self.shape)[0]

    @property
    def n_minor(self) -> int:
        return self._swap(self.shape)[1]

    @property
    def nnz(self) -> int:
        return int(self.indices.shape[0])

    @property
    def dtype(self) -> np.dtype:
        return self.values.dtype

    @property
    def n_unique(self) -> int:
        """Number of stored (major-slice, unique-value) entries."""
        return int(self.values.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        cls = type(self).__name__
        return (
            f"<{cls} shape={self.shape} dtype={self.dtype} "
            f"nnz={self.nnz} n_unique={self.n_unique}>"
        )

    def copy(self):
        return type(self)(
            self.shape,
            self.major_ptr.copy(),
            self.values.copy(),
            self.value_ptr.copy(),
            self.indices.copy(),
        )

    # -- construction / conversion ---------------------------------------

    @classmethod
    def from_scipy(cls, mat) -> _VCSBase:
        """Build from any scipy sparse array/matrix (converted internally)."""
        mat = mat.tocsc() if cls._format == "csc" else mat.tocsr()
        n_major, _n_minor = cls._swap(mat.shape)
        major_ptr, values, value_ptr, indices = _construct.compress(
            mat.indptr, mat.indices, mat.data, n_major
        )
        return cls(mat.shape, major_ptr, values, value_ptr, indices)

    def to_scipy(self):
        """Decompress to the equivalent scipy ``csc_array``/``csr_array``."""
        n_major, _n_minor = self._swap(self.shape)
        major_ptr, indices, data = _construct.decompress(
            self.major_ptr, self.values, self.value_ptr, self.indices, n_major
        )
        cls = sp.csc_array if self._format == "csc" else sp.csr_array
        return cls((data, indices, major_ptr), shape=self.shape)

    def to_csc(self) -> sp.csc_array:
        return self.to_scipy().tocsc()

    def to_csr(self) -> sp.csr_array:
        return self.to_scipy().tocsr()

    def toarray(self) -> np.ndarray:
        return self.to_scipy().toarray()

    # -- structural ops ----------------------------------------------------

    @property
    def T(self) -> _VCSBase:
        """Transpose. Free: shares buffers and swaps the VCSC/VCSR dual class."""
        other_cls = VCSRArray if self._format == "csc" else VCSCArray
        return other_cls(
            (self.shape[1], self.shape[0]),
            self.major_ptr,
            self.values,
            self.value_ptr,
            self.indices,
        )

    def transpose(self) -> _VCSBase:
        return self.T

    def log1p(self) -> _VCSBase:
        """Elementwise ``log1p``. Structural zeros stay zero implicitly."""
        return type(self)(
            self.shape,
            self.major_ptr.copy(),
            np.log1p(self.values),
            self.value_ptr.copy(),
            self.indices.copy(),
        )

    # -- reductions -----------------------------------------------------------

    def _major_sums(self) -> np.ndarray:
        """Per-major-slice totals -- cheap, doesn't touch ``indices``."""
        group_sizes = np.diff(self.value_ptr)
        weighted = self.values.astype(np.float64) * group_sizes
        group_of_major = np.repeat(np.arange(self.n_major, dtype=np.int64), np.diff(self.major_ptr))
        return np.bincount(group_of_major, weights=weighted, minlength=self.n_major)

    def _minor_sums(self) -> np.ndarray:
        """Per-minor-index totals -- a scatter-add over every nonzero."""
        group_sizes = np.diff(self.value_ptr)
        expanded = np.repeat(self.values.astype(np.float64), group_sizes)
        return np.bincount(self.indices, weights=expanded, minlength=self.n_minor)

    def sum(self, axis: int | None = None) -> np.ndarray | float:
        """Sum of (structural) values along ``axis`` (0=rows, 1=columns), or overall if ``None``."""
        if axis is None:
            group_sizes = np.diff(self.value_ptr)
            return float(np.sum(self.values.astype(np.float64) * group_sizes))
        if axis not in (0, 1):
            raise ValueError(f"axis must be None, 0, or 1, got {axis!r}")
        # axis=0 reduces over rows (column totals); axis=1 reduces over
        # columns (row totals). Major-slice totals are column totals for
        # VCSC (major=columns) and row totals for VCSR (major=rows).
        major_axis = 0 if self._format == "csc" else 1
        return self._major_sums() if axis == major_axis else self._minor_sums()

    # -- scalar arithmetic --------------------------------------------------

    def _empty_like(self) -> _VCSBase:
        n_major, _ = self._swap(self.shape)
        return type(self)(
            self.shape,
            np.zeros(n_major + 1, dtype=np.int64),
            np.empty(0, dtype=self.values.dtype),
            np.zeros(1, dtype=np.int64),
            np.empty(0, dtype=self.indices.dtype),
        )

    def __mul__(self, other):
        if not np.isscalar(other):
            return NotImplemented
        if other == 0:
            return self._empty_like()
        return type(self)(
            self.shape,
            self.major_ptr.copy(),
            self.values * other,
            self.value_ptr.copy(),
            self.indices.copy(),
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not np.isscalar(other):
            return NotImplemented
        return type(self)(
            self.shape,
            self.major_ptr.copy(),
            self.values / other,
            self.value_ptr.copy(),
            self.indices.copy(),
        )

    def __neg__(self):
        return self * -1

    # -- matrix products -----------------------------------------------------

    def _dot_right(self, x: np.ndarray) -> np.ndarray:
        """Compute ``self @ x`` for 1-D ``x`` of length ``n_cols``."""
        n_major, n_minor = self._swap(self.shape)
        args = (self.major_ptr, self.values, self.value_ptr, self.indices)
        if self._format == "csc":
            return _ops.major_matvec(*args, x, n_major, n_minor)
        return _ops.minor_matvec(*args, x, n_major)

    def _dot_left(self, x: np.ndarray) -> np.ndarray:
        """Compute ``x @ self`` for 1-D ``x`` of length ``n_rows``."""
        n_major, _n_minor = self._swap(self.shape)
        args = (self.major_ptr, self.values, self.value_ptr, self.indices)
        if self._format == "csc":
            return _ops.minor_matvec(*args, x, n_major)
        return _ops.major_matvec(*args, x, n_major, self.shape[1])

    def _dot_right_mat(self, b: np.ndarray) -> np.ndarray:
        n_major, n_minor = self._swap(self.shape)
        args = (self.major_ptr, self.values, self.value_ptr, self.indices)
        if self._format == "csc":
            return _ops.major_matmat(*args, b, n_major, n_minor)
        return _ops.minor_matmat(*args, b.T, n_major).T

    def _dot_left_mat(self, b: np.ndarray) -> np.ndarray:
        n_major, n_minor = self._swap(self.shape)
        args = (self.major_ptr, self.values, self.value_ptr, self.indices)
        if self._format == "csc":
            return _ops.minor_matmat(*args, b, n_major)
        return _ops.major_matmat(*args, b.T, n_major, n_minor).T

    def __matmul__(self, other):
        other_arr = np.asarray(other)
        if other_arr.ndim == 1:
            if other_arr.shape[0] != self.shape[1]:
                raise ValueError(
                    f"shapes {self.shape} and {other_arr.shape} not aligned"
                )
            return self._dot_right(other_arr)
        if other_arr.ndim == 2:
            if other_arr.shape[0] != self.shape[1]:
                raise ValueError(
                    f"shapes {self.shape} and {other_arr.shape} not aligned"
                )
            return self._dot_right_mat(other_arr)
        return NotImplemented

    def __rmatmul__(self, other):
        other_arr = np.asarray(other)
        if other_arr.ndim == 1:
            if other_arr.shape[0] != self.shape[0]:
                raise ValueError(
                    f"shapes {other_arr.shape} and {self.shape} not aligned"
                )
            return self._dot_left(other_arr)
        if other_arr.ndim == 2:
            if other_arr.shape[1] != self.shape[0]:
                raise ValueError(
                    f"shapes {other_arr.shape} and {self.shape} not aligned"
                )
            return self._dot_left_mat(other_arr)
        return NotImplemented

    # -- indexing -------------------------------------------------------------

    def _select_major(self, key: Any) -> _VCSBase:
        idx = _normalize_major_idx(key, self.n_major)
        starts = self.major_ptr[idx]
        ends = self.major_ptr[idx + 1]
        counts = ends - starts
        new_major_ptr = np.zeros(idx.shape[0] + 1, dtype=np.int64)
        np.cumsum(counts, out=new_major_ptr[1:])

        value_slots = np.concatenate(
            [np.arange(s, e) for s, e in zip(starts, ends, strict=True)]
        ) if idx.shape[0] else np.empty(0, dtype=np.int64)
        new_values = self.values[value_slots]

        v_starts = self.value_ptr[value_slots]
        v_ends = self.value_ptr[value_slots + 1]
        idx_counts = v_ends - v_starts
        new_value_ptr = np.zeros(value_slots.shape[0] + 1, dtype=np.int64)
        np.cumsum(idx_counts, out=new_value_ptr[1:])
        new_indices = np.concatenate(
            [self.indices[s:e] for s, e in zip(v_starts, v_ends, strict=True)]
        ) if value_slots.shape[0] else np.empty(0, dtype=self.indices.dtype)

        n_minor = self.n_minor
        new_shape = (
            (n_minor, idx.shape[0]) if self._format == "csc" else (idx.shape[0], n_minor)
        )
        return type(self)(new_shape, new_major_ptr, new_values, new_value_ptr, new_indices)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("VCSC/VCSR arrays are 2-D")
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)

        major_key, minor_key = (
            (col_key, row_key) if self._format == "csc" else (row_key, col_key)
        )
        if _is_full_slice(minor_key) and not _is_full_slice(major_key):
            return self._select_major(major_key)

        return self.to_scipy()[row_key, col_key]


class VCSCArray(_VCSBase):
    """Value-Compressed Sparse Column array. Values are deduplicated per column."""

    __slots__ = ()
    _format = "csc"


class VCSRArray(_VCSBase):
    """Value-Compressed Sparse Row array. Values are deduplicated per row."""

    __slots__ = ()
    _format = "csr"
