from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vcsc import VCSCArray, VCSRArray


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def _as_dense(result):
    return result.toarray() if hasattr(result, "toarray") else np.asarray(result)


def test_major_axis_slice(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    if vcls is VCSCArray:
        sub, expected = v[:, 1:3], dense[:, 1:3]
    else:
        sub, expected = v[1:3, :], dense[1:3, :]
    assert isinstance(sub, vcls)
    np.testing.assert_allclose(sub.to_scipy().toarray(), expected)


def test_major_axis_fancy_int(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    n = dense.shape[1] if vcls is VCSCArray else dense.shape[0]
    if n < 2:
        pytest.skip("axis too small")
    picks = [n - 1, 0]
    if vcls is VCSCArray:
        sub, expected = v[:, picks], dense[:, picks]
    else:
        sub, expected = v[picks, :], dense[picks, :]
    np.testing.assert_allclose(sub.to_scipy().toarray(), expected)


def test_major_axis_boolean(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    n = dense.shape[1] if vcls is VCSCArray else dense.shape[0]
    mask = np.zeros(n, dtype=bool)
    mask[::2] = True
    if vcls is VCSCArray:
        sub, expected = v[:, mask], dense[:, mask]
    else:
        sub, expected = v[mask, :], dense[mask, :]
    np.testing.assert_allclose(sub.to_scipy().toarray(), expected)


def test_general_2d_indexing_falls_back(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    if dense.shape[0] < 2 or dense.shape[1] < 2:
        pytest.skip("shape too small")
    result = v[0:2, 0:2]
    np.testing.assert_allclose(_as_dense(result), dense[0:2, 0:2])


def test_single_row_key_selects_rows(dense, vcls):
    v = vcls.from_scipy(sp.csr_array(dense))
    result = v[0]
    np.testing.assert_allclose(_as_dense(result).reshape(-1), dense[0].reshape(-1))
