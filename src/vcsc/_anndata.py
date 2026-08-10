"""Read/write helpers bridging :mod:`anndata` and the VCSC/VCSR layout."""

from __future__ import annotations

import anndata as ad
import scipy.sparse as sp

from vcsc._base import VCSCArray, VCSRArray, _VCSBase

__all__ = ["from_anndata", "to_layer"]


def _get_matrix(adata: ad.AnnData, layer: str | None, use_raw: bool):
    if use_raw:
        if adata.raw is None:
            raise ValueError("adata.raw is None; cannot use_raw=True")
        return adata.raw.X
    if layer is not None:
        return adata.layers[layer]
    return adata.X


def from_anndata(
    adata: ad.AnnData,
    layer: str | None = None,
    use_raw: bool = False,
    format: str = "csc",
) -> _VCSBase:
    """Convert ``adata.X`` (or a layer / ``raw.X``) into a VCSC/VCSR array.

    Parameters
    ----------
    adata
        Source :class:`anndata.AnnData` object.
    layer
        If given, read ``adata.layers[layer]`` instead of ``adata.X``.
    use_raw
        If ``True``, read ``adata.raw.X`` instead of ``adata.X``.
    format
        Either ``"csc"`` (default) or ``"csr"``, selecting the returned type.

    Returns
    -------
    A :class:`~vcsc.VCSCArray` or :class:`~vcsc.VCSRArray`. The source
    matrix may already be CSC, CSR, or dense; it is converted as needed.
    """
    if layer is not None and use_raw:
        raise ValueError("pass only one of layer= or use_raw=")
    if format not in ("csc", "csr"):
        raise ValueError(f"format must be 'csc' or 'csr', got {format!r}")

    mat = _get_matrix(adata, layer, use_raw)
    if not sp.issparse(mat):
        mat = sp.csr_array(mat)

    cls = VCSCArray if format == "csc" else VCSRArray
    return cls.from_scipy(mat)


def to_layer(adata: ad.AnnData, arr: _VCSBase, key: str) -> None:
    """Decompress ``arr`` and store it as ``adata.layers[key]``.

    AnnData does not natively understand the VCSC/VCSR layout, so this
    stores the equivalent scipy sparse array.
    """
    if arr.shape != adata.shape:
        raise ValueError(f"shape mismatch: array is {arr.shape}, adata is {adata.shape}")
    adata.layers[key] = arr.to_scipy()
