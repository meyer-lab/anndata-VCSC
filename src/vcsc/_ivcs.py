"""In-memory IVCSC/IVCSR array types.

Unlike :class:`~vcsc.VCSCArray`/:class:`~vcsc.VCSRArray` (whose ``indices``
is a plain int array), these keep the minor-axis indices byte-packed
(delta+varint encoded -- see :mod:`vcsc._ivcsc`) *in memory*, not just on
disk. That is the point of this type: an IVCSR-formatted file can be loaded
straight into one of these and stay compact, rather than paying to expand
``indices`` back out just to hold it in a plain ``AnnData``.

Scope is deliberately narrow for now (see GH issue #5): indexing/subsetting
(:meth:`~_IVCSBase.__getitem__`, staying byte-packed) and summing reads
along either axis (:meth:`~_IVCSBase.sum`), both without necessarily
decoding every index (see :meth:`~_IVCSBase._major_sums`). Anything else --
matrix products, elementwise arithmetic, ufuncs -- raises ``RuntimeError``;
call :meth:`~_IVCSBase.to_vcs` or :meth:`~_IVCSBase.to_scipy` first if you
need one of those. Decoding indices (:attr:`~_IVCSBase.indices`) is done at
most once per object and cached.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from vcsc import _construct, _ivcsc
from vcsc._indexutils import is_full_slice, normalize_major_idx

if TYPE_CHECKING:
    from vcsc._base import VCSCArray, VCSRArray, _VCSBase

__all__ = ["IVCSCArray", "IVCSRArray"]


class _IVCSBase:
    """Shared implementation for :class:`IVCSCArray` and :class:`IVCSRArray`."""

    _format: str

    __array_ufunc__ = None

    __slots__ = (
        "_indices_cache",
        "indices_dtype",
        "major_ptr",
        "packed_indices",
        "shape",
        "value_ptr",
        "values",
    )

    def __init__(
        self,
        shape: tuple[int, int],
        major_ptr: np.ndarray,
        values: np.ndarray,
        value_ptr: np.ndarray,
        packed_indices: np.ndarray,
        indices_dtype: np.dtype | type | str,
    ) -> None:
        n_major, _n_minor = self._swap(shape)
        major_ptr = np.asarray(major_ptr, dtype=np.int64)
        value_ptr = np.asarray(value_ptr, dtype=np.int64)
        values = np.asarray(values)
        packed_indices = np.asarray(packed_indices, dtype=np.uint8)

        if major_ptr.shape[0] != n_major + 1:
            raise ValueError(
                f"major_ptr has length {major_ptr.shape[0]}, expected {n_major + 1}"
            )
        if value_ptr.shape[0] != values.shape[0] + 1:
            raise ValueError("value_ptr must have length len(values) + 1")
        if major_ptr[-1] != values.shape[0]:
            raise ValueError("major_ptr[-1] must equal len(values)")

        self.shape = (int(shape[0]), int(shape[1]))
        self.major_ptr = major_ptr
        self.values = values
        self.value_ptr = value_ptr
        self.packed_indices = packed_indices
        self.indices_dtype = np.dtype(indices_dtype)
        self._indices_cache: np.ndarray | None = None

    # -- axis bookkeeping ------------------------------------------------

    @classmethod
    def _swap(cls, shape: tuple[int, int]) -> tuple[int, int]:
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
        return int(self.value_ptr[-1])

    @property
    def dtype(self) -> np.dtype:
        return self.values.dtype

    @property
    def n_unique(self) -> int:
        return int(self.values.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        cls = type(self).__name__
        return (
            f"<{cls} shape={self.shape} dtype={self.dtype} nnz={self.nnz} "
            f"n_unique={self.n_unique} packed_bytes={self.packed_indices.shape[0]}>"
        )

    def copy(self) -> _IVCSBase:
        return type(self)(
            self.shape,
            self.major_ptr.copy(),
            self.values.copy(),
            self.value_ptr.copy(),
            self.packed_indices.copy(),
            self.indices_dtype,
        )

    # -- decoding (lazy, cached) -------------------------------------------

    @property
    def indices(self) -> np.ndarray:
        """Decoded minor-axis indices. Decoded from ``packed_indices`` once, then cached."""
        if self._indices_cache is None:
            self._indices_cache = _ivcsc.unpack_indices(
                self.value_ptr, self.packed_indices, self.indices_dtype
            )
        return self._indices_cache

    # -- construction / conversion -------------------------------------------

    @classmethod
    def from_scipy(cls, mat) -> _IVCSBase:
        """Build from any scipy sparse array/matrix (converted internally)."""
        mat = mat.tocsc() if cls._format == "csc" else mat.tocsr()
        n_major, _n_minor = cls._swap(mat.shape)
        major_ptr, values, value_ptr, indices = _construct.compress(
            mat.indptr, mat.indices, mat.data, n_major
        )
        packed = _ivcsc.pack_indices(value_ptr, indices)
        return cls(mat.shape, major_ptr, values, value_ptr, packed, indices.dtype)

    @classmethod
    def from_vcs(cls, arr: _VCSBase) -> _IVCSBase:
        """Build from a matching-format :class:`~vcsc.VCSCArray`/:class:`~vcsc.VCSRArray`."""
        if arr._format != cls._format:
            raise ValueError(f"format mismatch: {type(arr).__name__} is not {cls._format!r}")
        packed = _ivcsc.pack_indices(arr.value_ptr, arr.indices)
        return cls(
            arr.shape,
            arr.major_ptr.copy(),
            arr.values.copy(),
            arr.value_ptr.copy(),
            packed,
            arr.indices.dtype,
        )

    def to_vcs(self) -> VCSCArray | VCSRArray:
        """Decode ``indices`` back to a plain array, as a VCSCArray/VCSRArray."""
        from vcsc._base import VCSCArray, VCSRArray

        cls = VCSCArray if self._format == "csc" else VCSRArray
        return cls(self.shape, self.major_ptr.copy(), self.values.copy(), self.value_ptr.copy(), self.indices.copy())

    def to_scipy(self):
        """Decompress to the equivalent scipy ``csc_array``/``csr_array``."""
        return self.to_vcs().to_scipy()

    def to_csc(self) -> sp.csc_array:
        return self.to_scipy().tocsc()

    def to_csr(self) -> sp.csr_array:
        return self.to_scipy().tocsr()

    def toarray(self) -> np.ndarray:
        return self.to_scipy().toarray()

    # -- structural ops ----------------------------------------------------

    @property
    def T(self) -> _IVCSBase:
        """Transpose. Free: shares buffers and swaps the IVCSC/IVCSR dual class."""
        other_cls = IVCSRArray if self._format == "csc" else IVCSCArray
        return other_cls(
            (self.shape[1], self.shape[0]),
            self.major_ptr,
            self.values,
            self.value_ptr,
            self.packed_indices,
            self.indices_dtype,
        )

    def transpose(self) -> _IVCSBase:
        return self.T

    # -- reductions -----------------------------------------------------------

    def _major_sums(self) -> np.ndarray:
        """Per-major-slice totals -- cheap, doesn't require decoding ``indices``."""
        group_sizes = np.diff(self.value_ptr)
        weighted = self.values.astype(np.float64) * group_sizes
        group_of_major = np.repeat(np.arange(self.n_major, dtype=np.int64), np.diff(self.major_ptr))
        return np.bincount(group_of_major, weights=weighted, minlength=self.n_major)

    def _minor_sums(self) -> np.ndarray:
        """Per-minor-index totals -- forces decoding ``indices``."""
        group_sizes = np.diff(self.value_ptr)
        expanded = np.repeat(self.values.astype(np.float64), group_sizes)
        return np.bincount(self.indices, weights=expanded, minlength=self.n_minor)

    def sum(self, axis: int | None = None) -> np.ndarray | float:
        """Sum of (structural) values along ``axis`` (0=rows, 1=columns), or overall if ``None``.

        Summing along the major axis (rows for IVCSR, columns for IVCSC)
        never decodes ``indices``; summing along the minor axis does.
        """
        if axis is None:
            group_sizes = np.diff(self.value_ptr)
            return float(np.sum(self.values.astype(np.float64) * group_sizes))
        if axis not in (0, 1):
            raise ValueError(f"axis must be None, 0, or 1, got {axis!r}")
        # axis=0 reduces over rows (column totals); axis=1 reduces over
        # columns (row totals). Major-slice totals are column totals for
        # IVCSC (major=columns) and row totals for IVCSR (major=rows).
        major_axis = 0 if self._format == "csc" else 1
        return self._major_sums() if axis == major_axis else self._minor_sums()

    # -- indexing -------------------------------------------------------------

    def _select_major(self, key: Any) -> _IVCSBase:
        """Major-axis-only slice. Decodes ``indices`` once, then repacks just the kept subset."""
        idx = normalize_major_idx(key, self.n_major)
        starts = self.major_ptr[idx]
        ends = self.major_ptr[idx + 1]
        counts = ends - starts
        new_major_ptr = np.zeros(idx.shape[0] + 1, dtype=np.int64)
        np.cumsum(counts, out=new_major_ptr[1:])

        value_slots = (
            np.concatenate([np.arange(s, e) for s, e in zip(starts, ends, strict=True)])
            if idx.shape[0]
            else np.empty(0, dtype=np.int64)
        )
        new_values = self.values[value_slots]

        v_starts = self.value_ptr[value_slots]
        v_ends = self.value_ptr[value_slots + 1]
        idx_counts = v_ends - v_starts
        new_value_ptr = np.zeros(value_slots.shape[0] + 1, dtype=np.int64)
        np.cumsum(idx_counts, out=new_value_ptr[1:])

        indices = self.indices
        new_indices = (
            np.concatenate([indices[s:e] for s, e in zip(v_starts, v_ends, strict=True)])
            if value_slots.shape[0]
            else np.empty(0, dtype=indices.dtype)
        )
        new_packed = _ivcsc.pack_indices(new_value_ptr, new_indices)

        n_minor = self.n_minor
        new_shape = (
            (n_minor, idx.shape[0]) if self._format == "csc" else (idx.shape[0], n_minor)
        )
        return type(self)(new_shape, new_major_ptr, new_values, new_value_ptr, new_packed, new_indices.dtype)

    def __getitem__(self, key) -> _IVCSBase:
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("IVCSC/IVCSR arrays are 2-D")
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)

        major_key, minor_key = (
            (col_key, row_key) if self._format == "csc" else (row_key, col_key)
        )
        if is_full_slice(minor_key) and not is_full_slice(major_key):
            return self._select_major(major_key)

        # General (both-axes) indexing: no shortcut exists that avoids
        # visiting every nonzero, so decompress, slice with scipy, and
        # repack -- the result is still an IVCSCArray/IVCSRArray either way.
        sub = self.to_scipy()[row_key, col_key]
        return type(self).from_scipy(sub)

    # -- explicitly-unsupported operations ------------------------------------

    def _unsupported(self, op: str) -> Any:
        raise RuntimeError(
            f"{op} is not supported on {type(self).__name__} (packed IVCSC/IVCSR arrays). "
            "Call .to_vcs() or .to_scipy() first if you need it."
        )

    def __matmul__(self, other: Any) -> Any:
        return self._unsupported("matrix multiplication")

    def __rmatmul__(self, other: Any) -> Any:
        return self._unsupported("matrix multiplication")

    def __mul__(self, other: Any) -> Any:
        return self._unsupported("multiplication")

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Any:
        return self._unsupported("division")

    def __add__(self, other: Any) -> Any:
        return self._unsupported("addition")

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:
        return self._unsupported("subtraction")

    def __rsub__(self, other: Any) -> Any:
        return self._unsupported("subtraction")

    def __neg__(self) -> Any:
        return self._unsupported("negation")


class IVCSCArray(_IVCSBase):
    """Value-Compressed Sparse Column array with byte-packed (IVCSC) indices."""

    __slots__ = ()
    _format = "csc"


class IVCSRArray(_IVCSBase):
    """Value-Compressed Sparse Row array with byte-packed (IVCSR) indices."""

    __slots__ = ()
    _format = "csr"
