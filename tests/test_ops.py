from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vcsc import VCSCArray, VCSRArray


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def _make(vcls, dense):
    return vcls.from_scipy(sp.csr_array(dense))


def test_scalar_mul(dense, vcls):
    v = _make(vcls, dense)
    for scalar in (2.0, -1.5, 0.0):
        out = v * scalar
        np.testing.assert_allclose(out.to_scipy().toarray(), dense * scalar)
        out2 = scalar * v
        np.testing.assert_allclose(out2.to_scipy().toarray(), dense * scalar)


def test_scalar_div(dense, vcls):
    v = _make(vcls, dense)
    out = v / 2.0
    np.testing.assert_allclose(out.to_scipy().toarray(), dense / 2.0)


def test_neg(dense, vcls):
    v = _make(vcls, dense)
    np.testing.assert_allclose((-v).to_scipy().toarray(), -dense)


def test_transpose(dense, vcls):
    v = _make(vcls, dense)
    vt = v.T
    np.testing.assert_allclose(vt.to_scipy().toarray(), dense.T)
    assert vt.shape == dense.T.shape
    vtt = vt.T
    assert type(vtt) is type(v)
    np.testing.assert_allclose(vtt.to_scipy().toarray(), dense)


def test_log1p(dense, vcls):
    v = _make(vcls, dense)
    out = v.log1p()
    np.testing.assert_allclose(out.to_scipy().toarray(), np.log1p(dense))


def test_matvec_right(dense, vcls, rng):
    v = _make(vcls, dense)
    x = rng.random(dense.shape[1])
    np.testing.assert_allclose(v @ x, dense @ x, atol=1e-8)


def test_matvec_left(dense, vcls, rng):
    v = _make(vcls, dense)
    x = rng.random(dense.shape[0])
    np.testing.assert_allclose(x @ v, x @ dense, atol=1e-8)


def test_matmat_right(dense, vcls, rng):
    v = _make(vcls, dense)
    b = rng.random((dense.shape[1], 3))
    np.testing.assert_allclose(v @ b, dense @ b, atol=1e-8)


def test_matmat_left(dense, vcls, rng):
    v = _make(vcls, dense)
    b = rng.random((3, dense.shape[0]))
    np.testing.assert_allclose(b @ v, b @ dense, atol=1e-8)


def test_matvec_dimension_mismatch_raises(dense, vcls):
    """Verify that dimension mismatch in matrix-vector product raises ValueError."""
    v = _make(vcls, dense)
    with pytest.raises(ValueError, match="not aligned"):
        v @ np.ones(dense.shape[1] + 1)


def test_scalar_mul_zero_returns_empty_like(dense, vcls):
    """Verify that multiplying by 0 produces an empty-like array preserving shape and dtype."""
    v = _make(vcls, dense)
    v0 = v * 0
    assert isinstance(v0, vcls)
    assert v0.shape == v.shape
    assert v0.dtype == v.dtype
    assert v0.nnz == 0
    assert v0.n_unique == 0
    np.testing.assert_allclose(v0.toarray(), np.zeros(v.shape))


def test_unsupported_scalar_operands_raise(dense, vcls):
    """Verify that non-scalar operand types in arithmetic operators return NotImplemented or raise TypeError."""
    v = _make(vcls, dense)
    assert v.__mul__([1, 2]) is NotImplemented
    assert v.__truediv__([1, 2]) is NotImplemented
    with pytest.raises(TypeError):
        _ = v * {"a": 1}
    with pytest.raises(TypeError):
        _ = v / {"a": 1}


def test_matmat_dimension_mismatch_raises(dense, vcls):
    """Verify that dimension mismatches in 2-D matrix products raise ValueError."""
    v = _make(vcls, dense)
    # Right matrix multiplication dimension mismatch
    with pytest.raises(ValueError, match="not aligned"):
        v @ np.ones((dense.shape[1] + 2, 4))

    # Left vector multiplication dimension mismatch
    with pytest.raises(ValueError, match="not aligned"):
        np.ones(dense.shape[0] + 2) @ v

    # Left matrix multiplication dimension mismatch
    with pytest.raises(ValueError, match="not aligned"):
        np.ones((4, dense.shape[0] + 2)) @ v


def test_unsupported_matmul_operands_raise(dense, vcls):
    """Verify that >2D array operands in matmul raise TypeError."""
    v = _make(vcls, dense)
    arr_3d = np.ones((dense.shape[1], 2, 2))
    with pytest.raises(TypeError):
        _ = v @ arr_3d
    with pytest.raises(TypeError):
        _ = arr_3d @ v

