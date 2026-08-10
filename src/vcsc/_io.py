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

import numpy as np
from anndata.compat import H5Group, ZarrGroup

from vcsc._base import VCSCArray, VCSRArray, _VCSBase

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anndata._io.specs.registry import Reader, Writer
    from anndata._types import GroupStorageType

__all__: list[str] = []

_ARRAY_KEYS = ("major_ptr", "values", "value_ptr", "indices")
_SPEC_VERSION = "0.1.0"


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


def _register() -> None:
    from anndata._io.specs.registry import _REGISTRY, IOSpec

    for store_type in (H5Group, ZarrGroup):
        for cls, spec_name in ((VCSCArray, "vcsc"), (VCSRArray, "vcsr")):
            spec = IOSpec(spec_name, _SPEC_VERSION)
            if not _REGISTRY.has_write(store_type, cls, frozenset()):
                _REGISTRY.register_write(store_type, cls, spec)(_write_vcs)
            if not _REGISTRY.has_read(store_type, spec, frozenset()):
                _REGISTRY.register_read(store_type, spec, frozenset())(_make_read_vcs(cls))


_register()
