from __future__ import annotations

from typing import cast

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from vcsc import VCSCAnnData, VCSCArray, VCSRArray


@pytest.fixture
def base_adata(dense) -> ad.AnnData:
    obj = ad.AnnData(X=sp.csr_array(dense))
    obj.raw = obj
    obj.obs["grp"] = [str(i % 2) for i in range(dense.shape[0])]
    obj.var["gene"] = [f"g{i}" for i in range(dense.shape[1])]
    return obj


def test_from_anndata_and_back(base_adata, dense):
    va = VCSCAnnData.from_anndata(base_adata)
    assert isinstance(va.X, VCSCArray)
    assert isinstance(va.raw_X, VCSCArray)
    assert va.shape == dense.shape

    back = va.to_anndata()
    assert type(back) is ad.AnnData
    np.testing.assert_allclose(sp.csr_array(back.X).toarray(), dense)
    assert back.raw is not None
    np.testing.assert_allclose(sp.csr_array(back.raw.X).toarray(), dense)
    assert list(back.obs["grp"]) == list(base_adata.obs["grp"])
    assert list(back.var["gene"]) == list(base_adata.var["gene"])


def test_from_anndata_csr_format(base_adata, dense):
    va = VCSCAnnData.from_anndata(base_adata, format="csr")
    assert isinstance(va.X, VCSRArray)
    np.testing.assert_allclose(va.X.to_scipy().toarray(), dense)


def test_from_anndata_no_raw(base_adata, dense):
    va = VCSCAnnData.from_anndata(base_adata, include_raw=False)
    assert va.raw_X is None


def test_constructor_rejects_non_vcs_x(dense):
    with pytest.raises(TypeError):
        VCSCAnnData(X=dense)


def test_constructor_rejects_non_vcs_raw_x(dense):
    v = VCSCArray.from_scipy(sp.csc_array(dense))
    with pytest.raises(TypeError):
        VCSCAnnData(X=v, raw_X=dense)


def test_constructor_rejects_raw_kwarg(dense):
    v = VCSCArray.from_scipy(sp.csc_array(dense))
    with pytest.raises(TypeError):
        VCSCAnnData(X=v, raw="not supported")


def test_x_setter_validates_shape(dense):
    v = VCSCArray.from_scipy(sp.csc_array(dense))
    va = VCSCAnnData(X=v)
    other_shape = (dense.shape[0] + 1, dense.shape[1])
    bad = VCSCArray.from_scipy(sp.csc_array(np.zeros(other_shape)))
    with pytest.raises(ValueError):
        va.X = bad


def test_h5ad_roundtrip(base_adata, dense, tmp_path):
    va = VCSCAnnData.from_anndata(base_adata)
    path = tmp_path / "test.h5ad"
    va.write_h5ad(path)

    read_back = VCSCAnnData.read_h5ad(path)
    assert isinstance(read_back.X, VCSCArray)
    assert isinstance(read_back.raw_X, VCSCArray)
    np.testing.assert_allclose(read_back.X.to_scipy().toarray(), dense)
    np.testing.assert_allclose(read_back.raw_X.to_scipy().toarray(), dense)
    assert list(read_back.obs["grp"]) == list(base_adata.obs["grp"])
    assert list(read_back.var["gene"]) == list(base_adata.var["gene"])


def test_h5ad_roundtrip_no_raw(base_adata, dense, tmp_path):
    va = VCSCAnnData.from_anndata(base_adata, include_raw=False)
    path = tmp_path / "test_noraw.h5ad"
    va.write_h5ad(path)

    read_back = VCSCAnnData.read_h5ad(path)
    assert read_back.raw_X is None
    assert read_back.X is not None
    np.testing.assert_allclose(read_back.X.to_scipy().toarray(), dense)


def test_zarr_roundtrip(base_adata, dense, tmp_path):
    va = VCSCAnnData.from_anndata(base_adata)
    path = tmp_path / "test.zarr"
    va.write_zarr(path)

    read_back = VCSCAnnData.read_zarr(path)
    assert isinstance(read_back.X, VCSCArray)
    assert isinstance(read_back.raw_X, VCSCArray)
    np.testing.assert_allclose(read_back.X.to_scipy().toarray(), dense)
    np.testing.assert_allclose(read_back.raw_X.to_scipy().toarray(), dense)


def test_uns_embedding_roundtrips_via_registry(base_adata, dense, tmp_path):
    """VCSCArray registered as an IO codec also works nested inside uns."""
    v = VCSCArray.from_scipy(sp.csc_array(dense))
    base_adata.uns["vcsc_extra"] = v
    path = tmp_path / "uns_test.h5ad"

    import h5py

    with h5py.File(path, "w") as f:
        ad.io.write_elem(f, "uns", dict(base_adata.uns))

    with h5py.File(path, "r") as f:
        read_back = cast(dict, ad.io.read_elem(f["uns"]))

    assert isinstance(read_back["vcsc_extra"], VCSCArray)
    np.testing.assert_allclose(read_back["vcsc_extra"].to_scipy().toarray(), dense)
