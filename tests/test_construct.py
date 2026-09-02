"""Tests for vcsc._construct.transpose_major: direct VCSC<->VCSR storage regrouping."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from anndata_sc import VCSCArray, VCSRArray


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def _scipy_for(vcls, dense):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


def test_transpose_major_matches_dense(dense, vcls):
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    dual = v._transpose_major()

    other_cls = VCSRArray if vcls is VCSCArray else VCSCArray
    assert isinstance(dual, other_cls)
    assert dual.shape == dense.shape
    np.testing.assert_allclose(dual.toarray(), dense)


def test_transpose_major_preserves_value_dedup(dense, vcls):
    """Same logical nonzeros, so nnz must match; unique-value counts may differ per axis."""
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    dual = v._transpose_major()
    assert dual.nnz == v.nnz


def test_transpose_major_all_zero(vcls):
    dense = np.zeros((6, 5))
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    dual = v._transpose_major()
    assert dual.nnz == 0
    np.testing.assert_allclose(dual.toarray(), dense)


def test_transpose_major_is_involutive(dense, vcls):
    """Transposing twice returns to the original format with the same matrix."""
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    back = v._transpose_major()._transpose_major()
    assert type(back) is type(v)
    np.testing.assert_allclose(back.toarray(), dense)
