"""Tests for VCSCArray and VCSRArray base functionality, invariants, and conversions."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vcsc import VCSCArray, VCSRArray


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    """Fixture providing both VCSCArray and VCSRArray classes."""
    return request.param


def test_constructor_validates_major_ptr_length(vcls):
    """Verify that constructor rejects a major_ptr with incorrect length."""
    # Shape is (4, 5). For CSC, n_major = 5 (needs major_ptr len 6); for CSR, n_major = 4 (needs len 5).
    shape = (4, 5)
    invalid_major_ptr = np.array([0, 0, 0])  # Length 3 is invalid for both
    values = np.array([], dtype=np.float64)
    value_ptr = np.array([0])
    indices = np.array([], dtype=np.int64)

    with pytest.raises(ValueError, match="major_ptr has length"):
        vcls(shape, invalid_major_ptr, values, value_ptr, indices)


def test_constructor_validates_value_ptr_length(vcls):
    """Verify that constructor rejects a value_ptr not matching len(values) + 1."""
    shape = (2, 3)
    n_major = 3 if vcls is VCSCArray else 2
    major_ptr = np.zeros(n_major + 1, dtype=np.int64)
    values = np.array([1.0, 2.0])
    invalid_value_ptr = np.array([0, 1])  # Length 2 != len(values) + 1 (3)
    indices = np.array([0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="value_ptr must have length len\\(values\\) \\+ 1"):
        vcls(shape, major_ptr, values, invalid_value_ptr, indices)


def test_constructor_validates_major_ptr_terminal(vcls):
    """Verify that constructor rejects major_ptr whose final entry does not equal len(values)."""
    shape = (2, 3)
    n_major = 3 if vcls is VCSCArray else 2
    major_ptr = np.zeros(n_major + 1, dtype=np.int64)
    major_ptr[-1] = 5  # Mismatch with len(values) == 2
    values = np.array([1.0, 2.0])
    value_ptr = np.array([0, 1, 2], dtype=np.int64)
    indices = np.array([0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="major_ptr\\[-1\\] must equal len\\(values\\)"):
        vcls(shape, major_ptr, values, value_ptr, indices)


def test_constructor_validates_value_ptr_terminal(vcls):
    """Verify that constructor rejects value_ptr whose final entry does not equal len(indices)."""
    shape = (2, 3)
    n_major = 3 if vcls is VCSCArray else 2
    major_ptr = np.array([0] * n_major + [2], dtype=np.int64)
    values = np.array([1.0, 2.0])
    value_ptr = np.array([0, 1, 5], dtype=np.int64)  # Mismatch with len(indices) == 2
    indices = np.array([0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="value_ptr\\[-1\\] must equal len\\(indices\\)"):
        vcls(shape, major_ptr, values, value_ptr, indices)


def test_constructor_validates_indices_bounds(vcls):
    """Verify that constructor rejects out-of-bounds minor axis indices."""
    shape = (2, 3)
    n_major = 3 if vcls is VCSCArray else 2
    n_minor = 2 if vcls is VCSCArray else 3
    major_ptr = np.array([0] * n_major + [1], dtype=np.int64)
    values = np.array([1.0])
    value_ptr = np.array([0, 1], dtype=np.int64)

    # Negative index
    invalid_neg_indices = np.array([-1], dtype=np.int64)
    with pytest.raises(ValueError, match="indices out of bounds"):
        vcls(shape, major_ptr, values, value_ptr, invalid_neg_indices)

    # Out of upper bound index
    invalid_high_indices = np.array([n_minor], dtype=np.int64)
    with pytest.raises(ValueError, match="indices out of bounds"):
        vcls(shape, major_ptr, values, value_ptr, invalid_high_indices)


def test_array_properties_and_repr(dense, vcls):
    """Test structural dimension properties, nonzeros count, unique count, and __repr__."""
    v = vcls.from_scipy(sp.csr_array(dense))
    n_major = dense.shape[1] if vcls is VCSCArray else dense.shape[0]
    n_minor = dense.shape[0] if vcls is VCSCArray else dense.shape[1]

    assert v.n_major == n_major
    assert v.n_minor == n_minor
    assert v.nnz == np.count_nonzero(dense)
    assert v.dtype == dense.dtype
    assert isinstance(v.n_unique, int)
    assert v.n_unique == v.values.shape[0]

    # Verify repr format
    r = repr(v)
    assert type(v).__name__ in r
    assert f"shape={v.shape}" in r
    assert f"nnz={v.nnz}" in r


def test_copy(dense, vcls):
    """Verify that arr.copy() creates a deep copy with independent underlying buffers."""
    v = vcls.from_scipy(sp.csr_array(dense))
    v_copy = v.copy()

    assert isinstance(v_copy, vcls)
    assert v_copy.shape == v.shape
    np.testing.assert_allclose(v_copy.to_scipy().toarray(), v.to_scipy().toarray())

    # Modifying copy buffers should not alter the original
    if v_copy.values.size > 0:
        v_copy.values[0] += 10.0
        assert v_copy.values[0] != v.values[0]


def test_transpose_method(dense, vcls):
    """Verify that arr.transpose() is equivalent to arr.T."""
    v = vcls.from_scipy(sp.csr_array(dense))
    vt = v.transpose()
    assert vt.shape == (v.shape[1], v.shape[0])
    np.testing.assert_allclose(vt.to_scipy().toarray(), v.T.to_scipy().toarray())


def test_conversion_methods(dense, vcls):
    """Verify to_csc, to_csr, and toarray conversion helper methods."""
    v = vcls.from_scipy(sp.csr_array(dense))

    csc_out = v.to_csc()
    assert isinstance(csc_out, sp.csc_array)
    np.testing.assert_allclose(csc_out.toarray(), dense)

    csr_out = v.to_csr()
    assert isinstance(csr_out, sp.csr_array)
    np.testing.assert_allclose(csr_out.toarray(), dense)

    dense_out = v.toarray()
    assert isinstance(dense_out, np.ndarray)
    np.testing.assert_allclose(dense_out, dense)
