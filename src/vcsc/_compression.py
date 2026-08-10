"""Default on-disk compression for arrays written via anndata's IO registry.

Blosc2+LZ4 gives a good default speed/ratio tradeoff and, unlike gzip, is
fast enough to leave on by default. It requires the ``hdf5plugin`` filter to
be registered for HDF5; zarr already ships blosc support via ``numcodecs``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["h5_dataset_kwargs", "zarr_dataset_kwargs"]


def h5_dataset_kwargs() -> dict[str, Any]:
    """``h5py.Group.create_dataset`` kwargs for Blosc2+LZ4 compression."""
    import hdf5plugin

    return dict(hdf5plugin.Blosc2(cname="lz4"))


def zarr_dataset_kwargs() -> dict[str, Any]:
    """``dataset_kwargs`` for Blosc+LZ4 compression, understood by anndata's zarr writers."""
    from anndata.compat import is_zarr_v2

    if is_zarr_v2():
        import numcodecs

        return {"compressor": numcodecs.Blosc(cname="lz4")}

    import zarr.codecs

    return {"compressor": zarr.codecs.BloscCodec(cname="lz4")}
