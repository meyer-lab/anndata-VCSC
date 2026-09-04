"""Normalized, mean-centered *views* of :class:`~vsparse.VCSCArray`/:class:`~vsparse.VCSRArray`.

:class:`VCSCArrayNormalized`/:class:`VCSRArrayNormalized` wrap a plain
(unpacked) VCSC/VCSR array and behave like the read-depth-normalized,
log-transformed, mean-centered matrix that :func:`vsparse._rapid_load.
load_and_normalize` builds -- without ever materializing it. See
:mod:`vsparse._norm_common` for the shared statistics/materialization logic
(:class:`~vsparse._norm_common.NormalizedViewBase`) and :mod:`vsparse._vcs_matmul`
for the matmul kernels these use (a direct per-nonzero walk over the
already-decoded ``indices`` array).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vsparse._norm_common import NormalizedViewBase

if TYPE_CHECKING:
    from vsparse._base import _VCSBase

__all__ = ["VCSCArrayNormalized", "VCSRArrayNormalized"]


class _VCSNormalizedBase(NormalizedViewBase):
    """Shared implementation for :class:`VCSCArrayNormalized`/:class:`VCSRArrayNormalized`."""

    __slots__ = ("_dual_arr",)

    def __init__(self, arr: _VCSBase) -> None:
        super().__init__(arr)
        # Opposite-format copy of `arr`, cached by vsparse._vcs_matmul when
        # regrouping the whole array fits one chunk's budget.
        self._dual_arr: _VCSBase | None = None


    def __matmul__(self, other: Any) -> Any:
        """``self @ other`` for a dense ``other`` -- see :mod:`vsparse._vcs_matmul`."""
        from vsparse._vcs_matmul import normalized_at_dense

        return normalized_at_dense(self, other)

    def __rmatmul__(self, other: Any) -> Any:
        """``other @ self`` for a dense ``other`` -- see :mod:`vsparse._vcs_matmul`."""
        from vsparse._vcs_matmul import dense_at_normalized

        return dense_at_normalized(self, other)


class VCSCArrayNormalized(_VCSNormalizedBase):
    """Read-depth-normalized, log-transformed, mean-centered view of a :class:`~vsparse.VCSCArray`."""

    __slots__ = ()
    _format = "csc"


class VCSRArrayNormalized(_VCSNormalizedBase):
    """Read-depth-normalized, log-transformed, mean-centered view of a :class:`~vsparse.VCSRArray`."""

    __slots__ = ()
    _format = "csr"
