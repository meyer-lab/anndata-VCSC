from __future__ import annotations

import numpy as np

from anndata_sc import VCSCArray, VCSRArray


def test_csc_roundtrip(dense, csc):
    v = VCSCArray.from_scipy(csc)
    back = v.to_scipy()
    assert back.format == "csc"
    assert back.shape == csc.shape
    np.testing.assert_allclose(back.toarray(), dense)
    np.testing.assert_allclose(v.to_csr().toarray(), dense)
    assert v.nnz == csc.nnz


def test_csr_roundtrip(dense, csr):
    v = VCSRArray.from_scipy(csr)
    back = v.to_scipy()
    assert back.format == "csr"
    assert back.shape == csr.shape
    np.testing.assert_allclose(back.toarray(), dense)
    np.testing.assert_allclose(v.to_csc().toarray(), dense)
    assert v.nnz == csr.nnz


def test_csc_from_csr_input(dense, csr):
    v = VCSCArray.from_scipy(csr)
    np.testing.assert_allclose(v.to_scipy().toarray(), dense)


def test_value_compression_deduplicates(rng):
    dense = np.zeros((10, 10))
    dense[:, 0] = 3.0  # ten repeats of the same value in one column
    dense[0, 1] = 7.0
    v = VCSCArray.from_scipy(__import__("scipy.sparse", fromlist=["csc_array"]).csc_array(dense))
    assert v.nnz == 11
    assert v.n_unique == 2  # one unique value per nonempty column


def test_empty_matrix_roundtrip():
    import scipy.sparse as sp

    dense = np.zeros((6, 4))
    v = VCSCArray.from_scipy(sp.csc_array(dense))
    assert v.nnz == 0
    np.testing.assert_allclose(v.to_scipy().toarray(), dense)
