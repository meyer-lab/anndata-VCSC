"""Tests for IVCSCArrayNormalized/IVCSRArrayNormalized: normalized IVCSC/IVCSR views."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vcsc import IVCSCArray, IVCSCArrayNormalized, IVCSRArray, IVCSRArrayNormalized


@pytest.fixture(params=[IVCSCArray, IVCSRArray])
def ivcls(request):
    return request.param


def _scipy_for(ivcls, dense):
    return sp.csc_array(dense) if ivcls is IVCSCArray else sp.csr_array(dense)


def _norm_cls(ivcls):
    return IVCSCArrayNormalized if ivcls is IVCSCArray else IVCSRArrayNormalized


def _reference(dense: np.ndarray) -> np.ndarray:
    """Read-depth normalize, log-transform, and mean-center a dense matrix directly."""
    row_totals = dense.sum(axis=1)
    row_scale = row_totals / np.median(row_totals)
    row_scale[row_scale == 0.0] = 1.0  # rows with no counts: avoid div-by-zero (unused otherwise)
    scaled = dense / row_scale[:, None]
    gene_scale = scaled.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(gene_scale > 0, scaled / gene_scale[None, :], 0.0)
    transformed = np.log10(1.0 + 1000.0 * normalized)
    return transformed - transformed.mean(axis=0, keepdims=True)


def test_toarray_matches_reference(dense, ivcls):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    nv = v.normalized()
    assert isinstance(nv, _norm_cls(ivcls))
    np.testing.assert_allclose(nv.toarray(), _reference(dense), atol=1e-8)


def test_getitem_matches_reference_block(dense, ivcls):
    if dense.sum() == 0 or dense.shape[0] < 2 or dense.shape[1] < 2:
        pytest.skip("shape too small or all-zero")
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    nv = v.normalized()
    ref = _reference(dense)

    sub = nv[0:2, 0:2]
    np.testing.assert_allclose(sub, ref[0:2, 0:2], atol=1e-8)


def test_getitem_single_row_and_col(dense, ivcls):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix")
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    nv = v.normalized()
    ref = _reference(dense)

    np.testing.assert_allclose(nv[0, :], ref[0:1, :], atol=1e-8)
    np.testing.assert_allclose(nv[:, 0], ref[:, 0:1], atol=1e-8)


def test_getitem_boolean_mask(dense, ivcls):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix")
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    nv = v.normalized()
    ref = _reference(dense)

    mask = np.zeros(dense.shape[0], dtype=bool)
    mask[::2] = True
    np.testing.assert_allclose(nv[mask, :], ref[mask, :], atol=1e-8)


def test_shape_and_dtype(dense, ivcls):
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    nv = v.normalized()
    assert nv.shape == dense.shape
    assert nv.dtype == np.float64


def test_wrong_format_raises(ivcls):
    other = IVCSRArray if ivcls is IVCSCArray else IVCSCArray
    dense = np.eye(3)
    other_arr = other.from_scipy(_scipy_for(other, dense))
    with pytest.raises(ValueError, match="format"):
        _norm_cls(ivcls)(other_arr)


@pytest.mark.parametrize(
    "op",
    [
        lambda nv: nv @ np.ones(nv.shape[1]),
        lambda nv: np.ones(nv.shape[0]) @ nv,
        lambda nv: nv + nv,
        lambda nv: nv * 2,
    ],
)
def test_unsupported_operations_raise_runtime_error(dense, ivcls, op):
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    nv = v.normalized()
    with pytest.raises(RuntimeError, match="not supported"):
        op(nv)


def test_all_zero_matrix_does_not_crash(ivcls):
    dense = np.zeros((5, 4))
    v = ivcls.from_scipy(_scipy_for(ivcls, dense))
    nv = v.normalized()
    np.testing.assert_allclose(nv.toarray(), np.zeros((5, 4)))
