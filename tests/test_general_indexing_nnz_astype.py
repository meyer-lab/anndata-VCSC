"""Tests for general (both-axes) indexing, getnnz/count_nonzero, and astype."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCArray, VCSRArray


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


# -- general (both-axes) indexing --------------------------------------------


def test_general_slice_both_axes_native(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    if dense.shape[0] < 2 or dense.shape[1] < 2:
        pytest.skip("shape too small")
    sub = v[0:2, 0:2]
    assert isinstance(sub, vcls)
    np.testing.assert_allclose(sub.toarray(), dense[0:2, 0:2])


def test_slice_and_fancy_combo(dense, vcls):
    if dense.shape[0] < 3 or dense.shape[1] < 3:
        pytest.skip("shape too small")
    v = vcls.from_scipy(sp.csr_array(dense))
    sub = v[1:3, [0, 2]]
    assert isinstance(sub, vcls)
    np.testing.assert_allclose(sub.toarray(), dense[1:3][:, [0, 2]])


def test_boolean_mask_both_axes(dense, vcls):
    if dense.shape[0] < 2 or dense.shape[1] < 2:
        pytest.skip("shape too small")
    v = vcls.from_scipy(sp.csr_array(dense))
    row_mask = np.zeros(dense.shape[0], dtype=bool)
    row_mask[::2] = True
    col_mask = np.zeros(dense.shape[1], dtype=bool)
    col_mask[1::2] = True
    sub = v[row_mask, :][:, col_mask]
    assert isinstance(sub, vcls)
    np.testing.assert_allclose(sub.toarray(), dense[row_mask][:, col_mask])

    sub2 = v[row_mask][:, col_mask]
    np.testing.assert_allclose(sub2.toarray(), dense[row_mask][:, col_mask])


def test_minor_axis_only_selection(dense, vcls):
    """Selecting only along the minor axis (major key is a full slice) now stays native."""
    v = vcls.from_scipy(sp.csr_array(dense))
    n_minor = dense.shape[0] if vcls is VCSCArray else dense.shape[1]
    if n_minor < 2:
        pytest.skip("axis too small")
    picks = [n_minor - 1, 0]
    if vcls is VCSCArray:
        sub, expected = v[picks, :], dense[picks, :]
    else:
        sub, expected = v[:, picks], dense[:, picks]
    assert isinstance(sub, vcls)
    np.testing.assert_allclose(sub.toarray(), expected)


def test_minor_axis_boolean(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    n_minor = dense.shape[0] if vcls is VCSCArray else dense.shape[1]
    mask = np.zeros(n_minor, dtype=bool)
    mask[::2] = True
    if vcls is VCSCArray:
        sub, expected = v[mask, :], dense[mask, :]
    else:
        sub, expected = v[:, mask], dense[:, mask]
    assert isinstance(sub, vcls)
    np.testing.assert_allclose(sub.toarray(), expected)


def test_minor_axis_empty_selection(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    if vcls is VCSCArray:
        sub = v[[], :]
        assert sub.shape == (0, dense.shape[1])
    else:
        sub = v[:, []]
        assert sub.shape == (dense.shape[0], 0)
    assert isinstance(sub, vcls)
    assert sub.nnz == 0
    assert sub.n_unique == 0


def test_both_full_slice_returns_copy(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    sub = v[:, :]
    assert isinstance(sub, vcls)
    assert sub is not v
    np.testing.assert_allclose(sub.toarray(), dense)


def test_both_int_still_returns_scalar(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    val = v[0, 0]
    assert np.isscalar(val) or isinstance(val, np.generic)
    assert val == dense[0, 0]


# -- getnnz / count_nonzero ----------------------------------------------------


def test_getnnz_overall(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    assert v.getnnz() == np.count_nonzero(dense)


def test_getnnz_per_axis(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    np.testing.assert_array_equal(v.getnnz(axis=0), np.count_nonzero(dense, axis=0))
    np.testing.assert_array_equal(v.getnnz(axis=1), np.count_nonzero(dense, axis=1))


def test_getnnz_invalid_axis(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    with pytest.raises(ValueError):
        v.getnnz(axis=2)


def test_count_nonzero(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    assert v.count_nonzero() == np.count_nonzero(dense)


def test_count_nonzero_excludes_explicit_zero_values():
    """count_nonzero must exclude stored-but-zero values, unlike nnz/getnnz."""
    dense = np.array([[1.0, 0.0], [0.0, 2.0]])
    for vcls in (VCSCArray, VCSRArray):
        v = vcls.from_scipy(sp.csr_array(dense))
        # Manually inject an explicit zero into the stored (unique) values,
        # which nnz/getnnz still counts as a stored element.
        zeroed_values = v.values.copy()
        zeroed_values[0] = 0.0
        v2 = vcls(v.shape, v.major_ptr, zeroed_values, v.value_ptr, v.indices)
        assert v2.getnnz() == v.nnz
        assert v2.count_nonzero() < v2.getnnz()


# -- astype ---------------------------------------------------------------------


def test_astype_casts_values(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    out = v.astype(np.float32)
    assert isinstance(out, vcls)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.toarray(), dense.astype(np.float32))
    # original is untouched
    assert v.dtype == dense.dtype


def test_astype_no_copy_same_dtype_returns_self(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    out = v.astype(v.dtype, copy=False)
    assert out is v


def test_astype_copy_true_same_dtype_returns_new_object(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    out = v.astype(v.dtype, copy=True)
    assert out is not v
    np.testing.assert_allclose(out.toarray(), dense)
