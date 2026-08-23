"""Tests for IVCSCArray/IVCSRArray: byte-packed, in-memory IVCSC/IVCSR arrays."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vcsc import IVCSCArray, IVCSRArray, VCSCArray, VCSRArray


@pytest.fixture(params=[IVCSCArray, IVCSRArray])
def ivcls(request):
    return request.param


def _scipy_for(ivcls, dense):
    return sp.csc_array(dense) if ivcls is IVCSCArray else sp.csr_array(dense)


def test_from_scipy_roundtrips(dense, ivcls):
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    np.testing.assert_allclose(v.toarray(), dense)
    np.testing.assert_allclose(v.to_scipy().toarray(), dense)


def test_indices_not_decoded_until_accessed(dense, ivcls):
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    assert v._indices_cache is None
    _ = v.indices
    assert v._indices_cache is not None
    np.testing.assert_array_equal(v.indices, v._indices_cache)


def test_major_axis_sum_does_not_decode(dense, ivcls):
    """Summing along the major axis is a documented no-decode fast path."""
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    major_axis = 0 if ivcls is IVCSCArray else 1
    result = v.sum(axis=major_axis)
    assert v._indices_cache is None
    np.testing.assert_allclose(result, dense.sum(axis=major_axis))


def test_minor_axis_sum(dense, ivcls):
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    minor_axis = 1 if ivcls is IVCSCArray else 0
    np.testing.assert_allclose(v.sum(axis=minor_axis), dense.sum(axis=minor_axis))


def test_total_sum(dense, ivcls):
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    assert v.sum() == pytest.approx(dense.sum())


def test_sum_invalid_axis_raises(dense, ivcls):
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    with pytest.raises(ValueError, match="axis must be"):
        v.sum(axis=2)


def test_major_axis_slice_stays_packed(dense, ivcls):
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    if ivcls is IVCSCArray:
        sub, expected = v[:, 1:3], dense[:, 1:3]
    else:
        sub, expected = v[1:3, :], dense[1:3, :]
    assert isinstance(sub, ivcls)
    np.testing.assert_allclose(sub.toarray(), expected)


def test_dual_axis_indexing_stays_ivcs_type(dense, ivcls):
    if dense.shape[0] < 2 or dense.shape[1] < 2:
        pytest.skip("shape too small")
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    sub = v[0:2, 0:2]
    assert isinstance(sub, ivcls)
    np.testing.assert_allclose(sub.toarray(), dense[0:2, 0:2])


def test_transpose_shares_buffers(dense, ivcls):
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    t = v.T
    other = IVCSRArray if ivcls is IVCSCArray else IVCSCArray
    assert isinstance(t, other)
    assert t.packed_indices is v.packed_indices
    np.testing.assert_allclose(t.toarray(), dense.T)


def test_to_vcs_and_from_vcs_roundtrip(dense, ivcls):
    vcls = VCSCArray if ivcls is IVCSCArray else VCSRArray
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    vcs = v.to_vcs()
    assert isinstance(vcs, vcls)
    np.testing.assert_allclose(vcs.to_scipy().toarray(), dense)

    back = ivcls.from_vcs(vcs)
    assert isinstance(back, ivcls)
    np.testing.assert_allclose(back.toarray(), dense)


def test_copy_is_independent(dense, ivcls):
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    if v.n_unique == 0:
        pytest.skip("no stored values to mutate")
    c = v.copy()
    c.values[:] = 0
    assert not np.array_equal(v.values, c.values)


@pytest.mark.parametrize(
    "op",
    [
        lambda v: v @ np.ones(v.shape[1]),
        lambda v: np.ones(v.shape[0]) @ v,
        lambda v: v * 2,
        lambda v: 2 * v,
        lambda v: v / 2,
        lambda v: v + v,
        lambda v: v - v,
        lambda v: -v,
    ],
)
def test_unsupported_operations_raise_runtime_error(dense, ivcls, op):
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    with pytest.raises(RuntimeError, match="not supported"):
        op(v)


def test_constructor_validates_major_ptr_length(ivcls):
    shape = (4, 5)
    invalid_major_ptr = np.array([0, 0, 0])
    values = np.array([], dtype=np.float64)
    value_ptr = np.array([0])
    packed = np.array([], dtype=np.uint8)
    with pytest.raises(ValueError, match="major_ptr has length"):
        ivcls(shape, invalid_major_ptr, values, value_ptr, packed, np.int64)
