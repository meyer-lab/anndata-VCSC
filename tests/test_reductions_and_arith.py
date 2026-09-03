"""Tests for per-axis sum/mean/max/min and elementwise arithmetic on VCSCArray/VCSRArray."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCArray, VCSRArray


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def make_signed_dense(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Dense matrix with negative, positive, and structural-zero entries."""
    dense = rng.integers(-5, 6, size=shape).astype(np.float64)
    mask = rng.random(shape) < 0.4
    dense[mask] = 0.0
    return dense


@pytest.fixture(params=[(1, 1), (5, 1), (1, 7), (8, 6), (25, 40), (50, 3)])
def shape(request) -> tuple[int, int]:
    return request.param


@pytest.fixture
def signed_dense(shape) -> np.ndarray:
    return make_signed_dense(np.random.default_rng(7), shape)


# -- sum / mean -------------------------------------------------------------


def test_sum_per_axis(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    assert v.sum() == pytest.approx(dense.sum())
    np.testing.assert_allclose(v.sum(axis=0), dense.sum(axis=0))
    np.testing.assert_allclose(v.sum(axis=1), dense.sum(axis=1))


def test_sum_invalid_axis(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    with pytest.raises(ValueError):
        v.sum(axis=2)


def test_mean_per_axis(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    assert v.mean() == pytest.approx(dense.mean())
    np.testing.assert_allclose(v.mean(axis=0), dense.mean(axis=0))
    np.testing.assert_allclose(v.mean(axis=1), dense.mean(axis=1))


# -- max / min ----------------------------------------------------------------


def test_max_per_axis(signed_dense, vcls):
    v = vcls.from_scipy(sp.csr_array(signed_dense))
    assert v.max() == pytest.approx(signed_dense.max())
    np.testing.assert_allclose(v.max(axis=0), signed_dense.max(axis=0))
    np.testing.assert_allclose(v.max(axis=1), signed_dense.max(axis=1))


def test_min_per_axis(signed_dense, vcls):
    v = vcls.from_scipy(sp.csr_array(signed_dense))
    assert v.min() == pytest.approx(signed_dense.min())
    np.testing.assert_allclose(v.min(axis=0), signed_dense.min(axis=0))
    np.testing.assert_allclose(v.min(axis=1), signed_dense.min(axis=1))


def test_max_min_invalid_axis(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    with pytest.raises(ValueError):
        v.max(axis=2)
    with pytest.raises(ValueError):
        v.min(axis=2)


def test_max_min_all_negative_column():
    """A column of only negative values must still report 0 if it has a structural zero."""
    dense = np.array([[-1.0, -2.0], [0.0, -3.0]])
    for vcls in (VCSCArray, VCSRArray):
        v = vcls.from_scipy(sp.csr_array(dense))
        np.testing.assert_allclose(v.max(axis=0), dense.max(axis=0))
        np.testing.assert_allclose(v.min(axis=0), dense.min(axis=0))


def test_max_min_fully_dense_negative():
    """A fully-dense negative row/column must not spuriously include 0."""
    dense = np.array([[-1.0, -2.0], [-4.0, -3.0]])
    for vcls in (VCSCArray, VCSRArray):
        v = vcls.from_scipy(sp.csr_array(dense))
        assert v.max() == pytest.approx(-1.0)
        np.testing.assert_allclose(v.max(axis=0), dense.max(axis=0))
        np.testing.assert_allclose(v.max(axis=1), dense.max(axis=1))


# -- elementwise arithmetic ---------------------------------------------------


def test_add_sub_vcs_vcs(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    other_dense = dense * 2
    other = vcls.from_scipy(sp.csr_array(other_dense))

    added = v + other
    assert isinstance(added, vcls)
    np.testing.assert_allclose(added.toarray(), dense + other_dense)

    subbed = v - other
    assert isinstance(subbed, vcls)
    np.testing.assert_allclose(subbed.toarray(), dense - other_dense)


def test_add_sub_dense(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    other_dense = np.ones_like(dense)

    added = v + other_dense
    np.testing.assert_allclose(np.asarray(added), dense + other_dense)

    radded = other_dense + v
    np.testing.assert_allclose(np.asarray(radded), other_dense + dense)

    subbed = v - other_dense
    np.testing.assert_allclose(np.asarray(subbed), dense - other_dense)

    rsubbed = other_dense - v
    np.testing.assert_allclose(np.asarray(rsubbed), other_dense - dense)


def test_add_sub_zero_scalar(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    np.testing.assert_allclose((v + 0).toarray(), dense)
    np.testing.assert_allclose((v - 0).toarray(), dense)
    np.testing.assert_allclose((0 - v).toarray(), -dense)


def test_add_nonzero_scalar_raises(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    with pytest.raises(NotImplementedError):
        v + 5
    with pytest.raises(NotImplementedError):
        v - 5
    with pytest.raises(NotImplementedError):
        5 - v


def test_multiply_elementwise(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    other_dense = dense + 1  # avoid trivially all-zero result
    other = vcls.from_scipy(sp.csr_array(other_dense))

    prod = v.multiply(other)
    assert isinstance(prod, vcls)
    np.testing.assert_allclose(prod.toarray(), dense * other_dense)

    prod_star = v * other
    assert isinstance(prod_star, vcls)
    np.testing.assert_allclose(prod_star.toarray(), dense * other_dense)

    prod_dense = v.multiply(other_dense)
    assert isinstance(prod_dense, vcls)
    np.testing.assert_allclose(prod_dense.toarray(), dense * other_dense)


def test_scalar_mul_div_unaffected(dense, vcls):
    """Existing scalar multiply/divide behavior must be preserved."""
    v = vcls.from_scipy(sp.csr_array(dense))
    np.testing.assert_allclose((v * 3).toarray(), dense * 3)
    np.testing.assert_allclose((3 * v).toarray(), dense * 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        np.testing.assert_allclose((v / 2).toarray(), dense / 2)
    assert isinstance(v * 0, vcls)
    np.testing.assert_allclose((v * 0).toarray(), np.zeros_like(dense))
