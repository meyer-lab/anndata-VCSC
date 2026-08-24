"""Normalized, mean-centered *views* of :class:`~vcsc.IVCSCArray`/:class:`~vcsc.IVCSRArray`.

:class:`IVCSCArrayNormalized`/:class:`IVCSRArrayNormalized` wrap a raw,
byte-packed IVCSC/IVCSR array and behave like the read-depth-normalized,
log-transformed, mean-centered matrix that :func:`vcsc._rapid_load.
load_and_normalize` builds -- without ever materializing it. See
:mod:`vcsc._norm_common` for the shared statistics/materialization logic
(:class:`~vcsc._norm_common.NormalizedViewBase`) and :mod:`vcsc._ivcs_matmul`
for the byte-stream-fused matmul kernels these use.
"""

from __future__ import annotations

from typing import Any

from vcsc._norm_common import NormalizedViewBase

__all__ = ["IVCSCArrayNormalized", "IVCSRArrayNormalized"]


class _IVCSNormalizedBase(NormalizedViewBase):
    """Shared implementation for :class:`IVCSCArrayNormalized`/:class:`IVCSRArrayNormalized`."""

    __slots__ = ()

    def __matmul__(self, other: Any) -> Any:
        """``self @ other`` for a dense ``other`` -- see :mod:`vcsc._ivcs_matmul`."""
        from vcsc._ivcs_matmul import normalized_at_dense

        return normalized_at_dense(self, other)

    def __rmatmul__(self, other: Any) -> Any:
        """``other @ self`` for a dense ``other`` -- see :mod:`vcsc._ivcs_matmul`."""
        from vcsc._ivcs_matmul import dense_at_normalized

        return dense_at_normalized(self, other)


class IVCSCArrayNormalized(_IVCSNormalizedBase):
    """Read-depth-normalized, log-transformed, mean-centered view of an :class:`~vcsc.IVCSCArray`."""

    __slots__ = ()
    _format = "csc"


class IVCSRArrayNormalized(_IVCSNormalizedBase):
    """Read-depth-normalized, log-transformed, mean-centered view of an :class:`~vcsc.IVCSRArray`."""

    __slots__ = ()
    _format = "csr"
