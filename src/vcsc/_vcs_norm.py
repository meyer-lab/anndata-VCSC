"""Normalized, mean-centered *views* of :class:`~vcsc.VCSCArray`/:class:`~vcsc.VCSRArray`.

:class:`VCSCArrayNormalized`/:class:`VCSRArrayNormalized` wrap a plain
(unpacked) VCSC/VCSR array and behave like the read-depth-normalized,
log-transformed, mean-centered matrix that :func:`vcsc._rapid_load.
load_and_normalize` builds -- without ever materializing it. See
:mod:`vcsc._norm_common` for the shared statistics/materialization logic
(:class:`~vcsc._norm_common.NormalizedViewBase`) and :mod:`vcsc._vcs_matmul`
for the matmul kernels these use (a direct per-nonzero walk over the
already-decoded ``indices`` array).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vcsc._norm_common import NormalizedViewBase

if TYPE_CHECKING:
    from vcsc._base import _VCSBase

__all__ = ["VCSCArrayNormalized", "VCSRArrayNormalized"]


class _VCSNormalizedBase(NormalizedViewBase):
    """Shared implementation for :class:`VCSCArrayNormalized`/:class:`VCSRArrayNormalized`."""

    __slots__ = ("_dual_arr",)

    def __init__(self, arr: _VCSBase) -> None:
        super().__init__(arr)
        # Lazily built, cached opposite-format copy of `arr` -- see
        # vcsc._vcs_matmul._get_dual. Built at most once per view, the
        # first time a matmul needs it in the direction `arr` isn't
        # major-aligned for.
        self._dual_arr: _VCSBase | None = None

    def __matmul__(self, other: Any) -> Any:
        """``self @ other`` for a dense ``other`` -- see :mod:`vcsc._vcs_matmul`."""
        from vcsc._vcs_matmul import normalized_at_dense

        return normalized_at_dense(self, other)

    def __rmatmul__(self, other: Any) -> Any:
        """``other @ self`` for a dense ``other`` -- see :mod:`vcsc._vcs_matmul`."""
        from vcsc._vcs_matmul import dense_at_normalized

        return dense_at_normalized(self, other)


class VCSCArrayNormalized(_VCSNormalizedBase):
    """Read-depth-normalized, log-transformed, mean-centered view of a :class:`~vcsc.VCSCArray`."""

    __slots__ = ()
    _format = "csc"


class VCSRArrayNormalized(_VCSNormalizedBase):
    """Read-depth-normalized, log-transformed, mean-centered view of a :class:`~vcsc.VCSRArray`."""

    __slots__ = ()
    _format = "csr"
