from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from vcsc import _ivcsc
from vcsc._base import VCSCArray


def test_pack_unpack_roundtrip(csc):
    v = VCSCArray.from_scipy(csc)
    packed = _ivcsc.pack_indices(v.value_ptr, v.indices)
    assert packed.dtype == np.uint8
    unpacked = _ivcsc.unpack_indices(v.value_ptr, packed, v.indices.dtype)

    # Order within a (major-slice, value) group is not preserved (packing
    # sorts each group), so compare group-wise as sets rather than arrays.
    for g in range(v.value_ptr.shape[0] - 1):
        start, end = v.value_ptr[g], v.value_ptr[g + 1]
        assert set(unpacked[start:end].tolist()) == set(v.indices[start:end].tolist())


def test_pack_unpack_empty():
    value_ptr = np.zeros(1, dtype=np.int64)
    indices = np.empty(0, dtype=np.int32)
    packed = _ivcsc.pack_indices(value_ptr, indices)
    assert packed.shape == (0,)
    unpacked = _ivcsc.unpack_indices(value_ptr, packed, np.dtype(np.int32))
    assert unpacked.shape == (0,)


def test_pack_unpack_large_indices():
    # Force multi-byte varints.
    dense = np.zeros((300, 2))
    dense[10, 0] = 1.0
    dense[299, 0] = 1.0
    v = VCSCArray.from_scipy(sp.csc_array(dense))
    packed = _ivcsc.pack_indices(v.value_ptr, v.indices)
    unpacked = _ivcsc.unpack_indices(v.value_ptr, packed, v.indices.dtype)
    np.testing.assert_array_equal(np.sort(unpacked), np.sort(v.indices))
