"""Default on-disk compression for arrays written via anndata's IO registry.

Blosc2+LZ4 gives a good default speed/ratio tradeoff and, unlike gzip, is
fast enough to leave on by default. It requires the ``hdf5plugin`` filter to
be registered for HDF5; zarr already ships blosc support via ``numcodecs``.

Blosc is only ever applied to *numeric* arrays. anndata applies whatever
``dataset_kwargs`` it's given uniformly to every array it writes, including
the small variable-length string arrays nested inside dataframes and
categoricals (category labels, ``obs``/``var`` string columns, the
``_index``). At least some HDF5 filter-plugin builds (seen in this
environment: h5py 3.16 / HDF5 2.0.0 / hdf5plugin's Blosc2) segfault
(``SIGFPE``) when the Blosc2 filter is applied to a variable-length-string
dataset. :func:`numeric_only_compression` patches the relevant
``create_dataset``/``create_array`` calls for the duration of a write so
string/object arrays always land uncompressed, regardless of what
``dataset_kwargs`` was passed in -- callers don't have to know about this.

Strings are left uncompressed rather than given a different codec: gzip and
lzf are safe on variable-length strings here, but the size problem is the
per-row string itself, not its compression. Encoding low-cardinality
columns as categoricals turns the per-row data numeric, which the existing
Blosc2 path then compresses -- see
:meth:`vsparse.VCSCAnnData.write_h5ad`'s ``convert_strings_to_categoricals``.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["h5_dataset_kwargs", "numeric_only_compression", "zarr_dataset_kwargs"]

_H5_COMPRESSION_KEYS = ("compression", "compression_opts")
_ZARR_V2_COMPRESSION_KEYS = ("compressor",)
_ZARR_V3_COMPRESSION_KEYS = ("compressors", "compressor")


def h5_dataset_kwargs() -> dict[str, Any]:
    """``h5py.Group.create_dataset`` kwargs for Blosc2+LZ4 compression."""
    import hdf5plugin

    return dict(hdf5plugin.Blosc2(cname="lz4"))


def _is_zarr_v2() -> bool:
    try:
        import zarr

        return getattr(zarr, "__version__", "").startswith("2.")
    except Exception:
        return False


def zarr_dataset_kwargs() -> dict[str, Any]:
    """``dataset_kwargs`` for Blosc+LZ4 compression, understood by anndata's zarr writers."""
    if _is_zarr_v2():
        import numcodecs

        return {"compressor": numcodecs.Blosc(cname="lz4")}

    import zarr.codecs

    return {"compressor": zarr.codecs.BloscCodec(cname="lz4")}


def _is_string_like(dtype: Any, data: Any) -> bool:
    """True if a dataset write looks like a variable-length/string/object array."""
    for candidate in (dtype, getattr(data, "dtype", None)):
        if candidate is None:
            continue
        try:
            import h5py

            if h5py.check_string_dtype(candidate) is not None:
                return True
        except Exception:
            # h5py's check only understands its own/numpy dtypes -- anything
            # else (e.g. zarr's own dtype objects) falls through to the
            # numpy-dtype and class-name checks below.
            pass
        try:
            if np.dtype(candidate).kind in ("O", "U", "S"):
                return True
        except TypeError:
            # Not numpy-dtype-like, e.g. zarr v3's VariableLengthUTF8() dtype object.
            if any(tag in type(candidate).__name__ for tag in ("UTF8", "String")):
                return True
    return False


@contextlib.contextmanager
def _numeric_only_h5() -> Iterator[None]:
    import h5py

    original = h5py.Group.create_dataset

    def patched(self: h5py.Group, name: str, *args: Any, **kwargs: Any) -> h5py.Dataset:
        data = kwargs.get("data", args[0] if args else None)
        if _is_string_like(kwargs.get("dtype"), data):
            for key in _H5_COMPRESSION_KEYS:
                kwargs.pop(key, None)
        return original(self, name, *args, **kwargs)

    h5py.Group.create_dataset = patched
    try:
        yield
    finally:
        h5py.Group.create_dataset = original


@contextlib.contextmanager
def _numeric_only_zarr() -> Iterator[None]:
    import zarr

    if _is_zarr_v2():
        zarr_group = cast(Any, zarr.Group)
        original_ds = getattr(zarr_group, "create_dataset", None)
        if original_ds is not None:
            keys = _ZARR_V2_COMPRESSION_KEYS

            def patched_ds(self: Any, name: str, **kwargs: Any) -> Any:
                if _is_string_like(kwargs.get("dtype"), kwargs.get("data")):
                    for key in keys:
                        kwargs[key] = None
                return original_ds(self, name, **kwargs)

            zarr_group.create_dataset = patched_ds
            try:
                yield
            finally:
                zarr_group.create_dataset = original_ds
            return

    original = zarr.Group.create_array
    keys = _ZARR_V3_COMPRESSION_KEYS

    def patched(self: zarr.Group, name: str, **kwargs: Any) -> Any:
        if _is_string_like(kwargs.get("dtype"), kwargs.get("data")):
            for key in keys:
                if key in kwargs:
                    kwargs[key] = None if key == "compressor" else ()
        return original(self, name, **kwargs)

    zarr.Group.create_array = patched
    try:
        yield
    finally:
        zarr.Group.create_array = original


def numeric_only_compression(store_kind: str) -> contextlib.AbstractContextManager[None]:
    """Scoped patch making Blosc compression apply to numeric arrays only.

    ``store_kind`` is ``"h5"`` or ``"zarr"``. See the module docstring for why
    this is needed. Use as ``with numeric_only_compression("h5"): ...`` around
    the write.
    """
    if store_kind == "h5":
        return _numeric_only_h5()
    if store_kind == "zarr":
        return _numeric_only_zarr()
    raise ValueError(f"store_kind must be 'h5' or 'zarr', got {store_kind!r}")
