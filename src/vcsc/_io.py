"""Registers VCSCArray/VCSRArray with anndata's on-disk IO registry.

This lets a VCSCArray/VCSRArray be written/read anywhere anndata's generic
element writer (``anndata.io.write_elem``/``read_elem``) is used -- e.g.
nested inside ``adata.uns``, or as ``X``/a custom top-level key on a
:class:`~vcsc.VCSCAnnData` -- and survive a round trip through HDF5 or zarr
without any manual (de)serialization.

This relies on ``anndata._io.specs.registry``, an internal (not yet public)
extension mechanism -- the same one anndata itself uses to support
dask/awkward/cupy arrays. It is not guaranteed stable across anndata
versions; if this module fails to import, it means anndata's registry
internals changed shape.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import anndata as ad
import numpy as np
from anndata.compat import H5Group, ZarrGroup

from vcsc import _ivcsc
from vcsc._base import VCSCArray, VCSRArray, _VCSBase
from vcsc._ivcs import IVCSCArray, IVCSRArray, _IVCSBase

if TYPE_CHECKING:
    from collections.abc import Mapping

    import h5py
    import zarr
    from anndata._io.specs.registry import Reader, Writer

    GroupStorageType = h5py.Group | zarr.Group

__all__ = ["write_ivcs_elem"]

_ARRAY_KEYS = ("major_ptr", "values", "value_ptr", "indices")
_SPEC_VERSION = "0.1.0"
_IVCS_SPEC_VERSION = "0.1.0"
_IVCS_SPEC_NAMES: dict[type[_VCSBase], str] = {VCSCArray: "ivcsc", VCSRArray: "ivcsr"}
_PACKED_SPEC_VERSION = "0.2.0"
_PACKED_SPEC_NAMES: dict[type[_IVCSBase], str] = {IVCSCArray: "ivcsc", IVCSRArray: "ivcsr"}
_PACKED_CLASSES_BY_NAME: dict[str, type[_IVCSBase]] = {v: k for k, v in _PACKED_SPEC_NAMES.items()}


def _write_vcs(
    f: GroupStorageType,
    k: str,
    v: _VCSBase,
    *,
    _writer: Writer,
    dataset_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> None:
    g = f.require_group(k)
    g.attrs["shape"] = v.shape
    for name in _ARRAY_KEYS:
        _writer.write_elem(g, name, getattr(v, name), dataset_kwargs=dataset_kwargs)


def _make_read_vcs(cls: type[_VCSBase]):
    def _read(elem: GroupStorageType, *, _reader: Reader) -> _VCSBase:
        shape_vals = [int(s) for s in np.asarray(elem.attrs["shape"]).tolist()]
        shape = (shape_vals[0], shape_vals[1])
        arrays = {
            name: cast(np.ndarray, _reader.read_elem(elem[name])) for name in _ARRAY_KEYS
        }
        return cls(
            shape,
            arrays["major_ptr"],
            arrays["values"],
            arrays["value_ptr"],
            arrays["indices"],
        )

    return _read


def write_ivcs_elem(
    f: GroupStorageType,
    k: str,
    v: _VCSBase | _IVCSBase,
    *,
    dataset_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> None:
    """Write ``v`` in the byte-packed IVCSC/IVCSR on-disk format.

    This trades write/read cost for a smaller file: ``indices`` is delta+
    varint packed via :mod:`vcsc._ivcsc` instead of stored as a plain int
    array. It is meant for archival storage only -- reading it back (through
    the registry, e.g. via ``anndata.io.read_elem``) always reconstructs a
    normal VCSCArray/VCSRArray with a plain ``indices`` array.

    Unlike the VCSC/VCSR codec registered below, this is not reachable
    through anndata's generic ``write_elem`` type dispatch (which can only
    route by the Python type of ``v``, and both formats share the same
    VCSCArray/VCSRArray types) -- call it directly, e.g. from
    :meth:`~vcsc.VCSCAnnData.write_h5ad`.
    """
    g = f.require_group(k)
    g.attrs["shape"] = v.shape
    g.attrs["indices_dtype"] = np.dtype(v.indices.dtype).name
    for name in ("major_ptr", "values", "value_ptr"):
        ad.io.write_elem(g, name, getattr(v, name), dataset_kwargs=dataset_kwargs)
    packed = _ivcsc.pack_indices(v.value_ptr, v.indices)
    ad.io.write_elem(g, "packed_indices", packed, dataset_kwargs=dataset_kwargs)
    g.attrs["encoding-type"] = "ivcsc" if v._format == "csc" else "ivcsr"
    g.attrs["encoding-version"] = _IVCS_SPEC_VERSION


def _make_read_ivcs(cls: type[_VCSBase]):
    def _read(elem: GroupStorageType, *, _reader: Reader) -> _VCSBase:
        shape_vals = [int(s) for s in np.asarray(elem.attrs["shape"]).tolist()]
        shape = (shape_vals[0], shape_vals[1])
        major_ptr = cast(np.ndarray, _reader.read_elem(elem["major_ptr"]))
        values = cast(np.ndarray, _reader.read_elem(elem["values"]))
        value_ptr = cast(np.ndarray, _reader.read_elem(elem["value_ptr"]))
        packed = cast(np.ndarray, _reader.read_elem(elem["packed_indices"]))
        dtype = np.dtype(elem.attrs["indices_dtype"])
        indices = _ivcsc.unpack_indices(value_ptr, packed, dtype)
        return cls(shape, major_ptr, values, value_ptr, indices)

    return _read


def read_ivcs_elem_packed(elem: GroupStorageType, cls: type[_IVCSBase]) -> _IVCSBase:
    """Read a group written by :func:`write_ivcs_elem` straight into an IVCSCArray/IVCSRArray.

    Unlike ``ad.io.read_elem`` on the same group (which always reconstructs
    a plain VCSCArray/VCSRArray by decoding ``packed_indices``, see
    :func:`_make_read_ivcs`), this never decodes -- the packed bytes are
    copied as-is into the returned array. Used by
    :func:`~vcsc.load_packed` to load an on-disk IVCSR/IVCSC ``.h5ad``
    straight into a byte-packed, in-memory form.
    """
    shape_vals = [int(s) for s in np.asarray(elem.attrs["shape"]).tolist()]
    shape = (shape_vals[0], shape_vals[1])
    major_ptr = cast(np.ndarray, ad.io.read_elem(elem["major_ptr"]))
    values = cast(np.ndarray, ad.io.read_elem(elem["values"]))
    value_ptr = cast(np.ndarray, ad.io.read_elem(elem["value_ptr"]))
    packed = cast(np.ndarray, ad.io.read_elem(elem["packed_indices"]))
    dtype = np.dtype(elem.attrs["indices_dtype"])
    return cls(shape, major_ptr, values, value_ptr, packed, dtype)


def _write_packed(
    f: GroupStorageType,
    k: str,
    v: _IVCSBase,
    *,
    _writer: Writer,
    dataset_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> None:
    g = f.require_group(k)
    g.attrs["shape"] = v.shape
    g.attrs["indices_dtype"] = np.dtype(v.indices_dtype).name
    for name in ("major_ptr", "values", "value_ptr"):
        _writer.write_elem(g, name, getattr(v, name), dataset_kwargs=dataset_kwargs)
    _writer.write_elem(g, "packed_indices", v.packed_indices, dataset_kwargs=dataset_kwargs)


def _make_read_packed(cls: type[_IVCSBase]):
    def _read(elem: GroupStorageType, *, _reader: Reader) -> _IVCSBase:
        shape_vals = [int(s) for s in np.asarray(elem.attrs["shape"]).tolist()]
        shape = (shape_vals[0], shape_vals[1])
        major_ptr = cast(np.ndarray, _reader.read_elem(elem["major_ptr"]))
        values = cast(np.ndarray, _reader.read_elem(elem["values"]))
        value_ptr = cast(np.ndarray, _reader.read_elem(elem["value_ptr"]))
        packed = cast(np.ndarray, _reader.read_elem(elem["packed_indices"]))
        dtype = np.dtype(elem.attrs["indices_dtype"])
        return cls(shape, major_ptr, values, value_ptr, packed, dtype)

    return _read


def _register() -> None:
    from anndata._io.specs.registry import _REGISTRY, IOSpec

    for store_type in (H5Group, ZarrGroup):
        for cls, spec_name in ((VCSCArray, "vcsc"), (VCSRArray, "vcsr")):
            spec = IOSpec(spec_name, _SPEC_VERSION)
            if not _REGISTRY.has_write(store_type, cls, frozenset()):
                _REGISTRY.register_write(store_type, cls, spec)(_write_vcs)
            if not _REGISTRY.has_read(store_type, spec, frozenset()):
                _REGISTRY.register_read(store_type, spec, frozenset())(_make_read_vcs(cls))
        for cls, spec_name in _IVCS_SPEC_NAMES.items():
            spec = IOSpec(spec_name, _IVCS_SPEC_VERSION)
            if not _REGISTRY.has_read(store_type, spec, frozenset()):
                _REGISTRY.register_read(store_type, spec, frozenset())(_make_read_ivcs(cls))
        for cls, spec_name in _PACKED_SPEC_NAMES.items():
            spec = IOSpec(spec_name, _PACKED_SPEC_VERSION)
            if not _REGISTRY.has_write(store_type, cls, frozenset()):
                _REGISTRY.register_write(store_type, cls, spec)(_write_packed)
            if not _REGISTRY.has_read(store_type, spec, frozenset()):
                _REGISTRY.register_read(store_type, spec, frozenset())(_make_read_packed(cls))


_register()
