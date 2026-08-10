from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import vcsc
from vcsc import VCSCArray, VCSRArray


@pytest.fixture
def adata(dense) -> ad.AnnData:
    obj = ad.AnnData(X=sp.csr_array(dense))
    obj.raw = obj
    obj.layers["dense_layer"] = sp.csc_array(dense)
    return obj


def test_from_anndata_x_csc(adata, dense):
    v = vcsc.from_anndata(adata, format="csc")
    assert isinstance(v, VCSCArray)
    np.testing.assert_allclose(v.to_scipy().toarray(), dense)


def test_from_anndata_x_csr(adata, dense):
    v = vcsc.from_anndata(adata, format="csr")
    assert isinstance(v, VCSRArray)
    np.testing.assert_allclose(v.to_scipy().toarray(), dense)


def test_from_anndata_raw(adata, dense):
    v = vcsc.from_anndata(adata, use_raw=True)
    np.testing.assert_allclose(v.to_scipy().toarray(), dense)


def test_from_anndata_layer(adata, dense):
    v = vcsc.from_anndata(adata, layer="dense_layer")
    np.testing.assert_allclose(v.to_scipy().toarray(), dense)


def test_from_anndata_rejects_layer_and_raw(adata):
    with pytest.raises(ValueError):
        vcsc.from_anndata(adata, layer="dense_layer", use_raw=True)


def test_to_layer_roundtrip(adata, dense):
    v = vcsc.from_anndata(adata)
    vcsc.to_layer(adata, v, "vcsc_out")
    np.testing.assert_allclose(adata.layers["vcsc_out"].toarray(), dense)


def test_from_anndata_dense_x(dense):
    obj = ad.AnnData(X=dense)
    v = vcsc.from_anndata(obj)
    np.testing.assert_allclose(v.to_scipy().toarray(), dense)
