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
    v = _make(vcls, dense)
    with pytest.raises(ValueError):
        v @ np.ones(dense.shape[1] + 1)
