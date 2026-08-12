"""An AnnData subclass whose X (and optionally raw X) is VCSC/VCSR-backed."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import anndata as ad
import pandas as pd

from vcsc import _compression, _io
from vcsc._base import VCSCArray, VCSRArray, _VCSBase

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

__all__ = ["VCSCAnnData"]

_VCS_TYPES = (VCSCArray, VCSRArray)
_DF_KEYS = ("obs", "var")
_MAPPING_KEYS = ("obsm", "varm", "obsp", "varp", "layers", "uns")
_FIELD_KEYS = (*_DF_KEYS, *_MAPPING_KEYS)
_STORE_FORMATS = ("vcsc", "ivcsc")


def _check_vcs_type(value: Any, name: str) -> None:
    if value is not None and not isinstance(value, _VCS_TYPES):
        raise TypeError(
            f"{name} must be a VCSCArray or VCSRArray, got {type(value).__name__}. "
            f"Build one with VCSCArray.from_scipy(...) or vcsc.from_anndata(...)."
        )


class VCSCAnnData(ad.AnnData):
    """An :class:`~anndata.AnnData` whose ``X`` is a VCSCArray/VCSRArray.

    Standard :class:`~anndata.AnnData` validates every array assigned to
    ``X``/``layers``/etc. against a fixed allowlist of types (dense/sparse/
    dask), so a plain ``AnnData`` cannot hold a :class:`~vcsc.VCSCArray`
    directly. This subclass overrides the ``X`` property to store one
    without going through that validation. A "raw" VCSC/VCSR matrix, if any,
    is kept as ``.raw_X`` -- a plain attribute, *not* wired into anndata's own
    ``.raw``/``Raw`` machinery, which has the same restriction.

    Because of this, operations that need anndata's normal per-element type
    dispatch on ``X`` -- slicing into views, concatenation, most of
    scanpy/anndata's ecosystem -- are **not** supported while ``X`` is
    VCSC/VCSR-backed. Call :meth:`to_anndata` first to get a fully-featured,
    ordinary ``AnnData``.

    Persist with :meth:`write_h5ad`/:meth:`write_zarr` and
    :meth:`read_h5ad`/:meth:`read_zarr` (not the top-level
    ``anndata.read_h5ad``/``read_zarr``, which always reconstruct a plain
    ``AnnData`` and would fail validating a VCSC-typed ``X``).
    """

    def __init__(
        self,
        X: _VCSBase | None = None,
        *,
        raw_X: _VCSBase | None = None,
        **kwargs: Any,
    ) -> None:
        _check_vcs_type(X, "X")
        _check_vcs_type(raw_X, "raw_X")
        if "raw" in kwargs:
            raise TypeError(
                "VCSCAnnData does not support the standard `raw=` argument; pass `raw_X=` instead."
            )
        shape = X.shape if X is not None else kwargs.pop("shape", None)
        super().__init__(X=None, shape=shape, **kwargs)
        self._vcs_X: _VCSBase | None = X
        self._vcs_raw_X: _VCSBase | None = raw_X

    # -- X / raw_X ------------------------------------------------------------

    @property
    def X(self) -> _VCSBase | None:
        return self._vcs_X

    @X.setter
    def X(self, value: _VCSBase | None) -> None:
        _check_vcs_type(value, "X")
        if value is not None and value.shape != self.shape:
            raise ValueError(f"X shape {value.shape} does not match adata shape {self.shape}")
        self._vcs_X = value

    @property
    def raw_X(self) -> _VCSBase | None:
        """The raw/X matrix, as a VCSCArray/VCSRArray (see class docstring)."""
        return self._vcs_raw_X

    @raw_X.setter
    def raw_X(self, value: _VCSBase | None) -> None:
        _check_vcs_type(value, "raw_X")
        self._vcs_raw_X = value

    # -- conversion -------------------------------------------------------------

    @classmethod
    def from_anndata(
        cls,
        adata: ad.AnnData,
        format: str = "csc",
        raw_format: str | None = None,
        *,
        include_raw: bool = True,
    ) -> VCSCAnnData:
        """Build from a regular :class:`~anndata.AnnData`, compressing X (and raw.X)."""
        vcls = VCSCArray if format == "csc" else VCSRArray
        X = vcls.from_scipy(adata.X) if adata.X is not None else None
        raw_X = None
        if include_raw and adata.raw is not None:
            rcls = VCSCArray if (raw_format or format) == "csc" else VCSRArray
            raw_X = rcls.from_scipy(adata.raw.X)
        return cls(
            X=X,
            raw_X=raw_X,
            obs=cast(pd.DataFrame, adata.obs).copy(),
            var=cast(pd.DataFrame, adata.var).copy(),
            uns=adata.uns,
            obsm=dict(adata.obsm),
            varm=dict(adata.varm),
            obsp=dict(adata.obsp),
            varp=dict(adata.varp),
            layers=dict(adata.layers),
        )

    def to_anndata(self) -> ad.AnnData:
        """Decompress to a regular, fully-featured :class:`~anndata.AnnData`."""
        obs = cast(pd.DataFrame, self.obs)
        var = cast(pd.DataFrame, self.var)
        out = ad.AnnData(
            X=self._vcs_X.to_scipy() if self._vcs_X is not None else None,
            obs=obs.copy(),
            var=var.copy(),
            uns=self.uns,
            obsm=dict(self.obsm),
            varm=dict(self.varm),
            obsp=dict(self.obsp),
            varp=dict(self.varp),
            layers=dict(self.layers),
        )
        if self._vcs_raw_X is not None:
            out.raw = ad.AnnData(X=self._vcs_raw_X.to_scipy(), obs=obs.copy(), var=var.copy())
        return out

    # -- persistence --------------------------------------------------------------
    #
    # Implemented by hand (field-by-field, via anndata.io.write_elem/read_elem)
    # rather than delegating to anndata.write_h5ad/write_zarr/read_h5ad/read_zarr:
    # those either construct a plain AnnData (crashes coerce_array on a VCSC-typed
    # X) or dispatch on the exact Python type of `self` (VCSCAnnData isn't
    # registered as an "anndata"-encoded type, only AnnData is).

    def _write_group(
        self,
        g: Any,
        *,
        format: str = "vcsc",
        dataset_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ) -> None:
        if format not in _STORE_FORMATS:
            raise ValueError(f"format must be one of {_STORE_FORMATS}, got {format!r}")
        write_array = ad.io.write_elem if format == "vcsc" else _io.write_ivcs_elem
        if self._vcs_X is not None:
            write_array(g, "X", self._vcs_X, dataset_kwargs=dataset_kwargs)
        if self._vcs_raw_X is not None:
            write_array(g, "raw_X", self._vcs_raw_X, dataset_kwargs=dataset_kwargs)
        for key in _DF_KEYS:
            ad.io.write_elem(g, key, getattr(self, key), dataset_kwargs=dataset_kwargs)
        for key in _MAPPING_KEYS:
            ad.io.write_elem(g, key, dict(getattr(self, key)), dataset_kwargs=dataset_kwargs)
        g.attrs["encoding-type"] = "anndata"
        g.attrs["encoding-version"] = "0.1.0"

    @classmethod
    def _read_group(cls, g: Any) -> VCSCAnnData:
        # X/raw_X read back as plain VCSCArray/VCSRArray regardless of whether
        # they were stored "vcsc" or "ivcsc" -- the registry dispatches on the
        # encoding-type attr each group was written with, not on how it's read.
        kwargs = {k: ad.io.read_elem(g[k]) for k in _FIELD_KEYS if k in g}
        X = cast("_VCSBase | None", ad.io.read_elem(g["X"]) if "X" in g else None)
        raw_X = cast("_VCSBase | None", ad.io.read_elem(g["raw_X"]) if "raw_X" in g else None)
        return cls(X=X, raw_X=raw_X, **kwargs)

    def write_h5ad(
        self,
        filename: str | PathLike[str],
        *,
        format: str = "vcsc",
        dataset_kwargs: Mapping[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        """Write to ``.h5ad``. Read back with :meth:`read_h5ad`.

        Parameters
        ----------
        format
            ``"vcsc"`` (default) stores ``X``/``raw_X`` with plain int arrays
            for the minor-axis indices. ``"ivcsc"`` (IVCSC/IVCSR) instead
            byte-packs them (delta + varint encoding) for a smaller file, at
            the cost of extra work on write/read. Either way, ``X``/``raw_X``
            come back from :meth:`read_h5ad` as ordinary VCSCArray/VCSRArray
            objects -- ``"ivcsc"`` is purely an on-disk storage format.
        dataset_kwargs
            Passed to ``h5py.Group.create_dataset`` for every array written.
            Defaults to Blosc2+LZ4 compression; pass ``{}`` to store
            uncompressed. Either way, compression is only ever applied to
            numeric arrays -- see :func:`vcsc._compression.numeric_only_compression`.
        """
        import h5py

        if dataset_kwargs is None:
            dataset_kwargs = _compression.h5_dataset_kwargs()
        with (
            _compression.numeric_only_compression("h5"),
            h5py.File(filename, "w") as f,
        ):
            self._write_group(f, format=format, dataset_kwargs=dataset_kwargs)

    @classmethod
    def read_h5ad(cls, filename: str | PathLike[str]) -> VCSCAnnData:
        """Read a file written by :meth:`write_h5ad`."""
        import h5py

        with h5py.File(filename, "r") as f:
            return cls._read_group(f)

    def write_zarr(
        self,
        store: Any,
        *,
        format: str = "vcsc",
        dataset_kwargs: Mapping[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        """Write to a zarr store. Read back with :meth:`read_zarr`.

        See :meth:`write_h5ad` for ``format``/``dataset_kwargs`` (including the
        numeric-only compression behavior); the default compression here is
        Blosc+LZ4 via ``numcodecs``.
        """
        import zarr

        if dataset_kwargs is None:
            dataset_kwargs = _compression.zarr_dataset_kwargs()
        with _compression.numeric_only_compression("zarr"):
            f = zarr.open_group(store, mode="w")
            self._write_group(f, format=format, dataset_kwargs=dataset_kwargs)

    @classmethod
    def read_zarr(cls, store: Any) -> VCSCAnnData:
        """Read a store written by :meth:`write_zarr`."""
        import zarr

        f = zarr.open_group(store, mode="r")
        return cls._read_group(f)
