"""Index arrays are sized by the axis they address, not by the array beside them.

Covers both halves of that rule: ``indices`` narrowed at construction/write
time (so a small gene axis never costs int64), and ``_filter_and_compact``
choosing its pointer and column-index dtypes from separate bounds.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCAnnData, VCSCArray, VCSRArray
from vsparse._indexutils import smallest_index_dtype
from vsparse._rapid_load import _filter_and_compact

INT32_MAX = np.iinfo(np.int32).max


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def _scipy_for(vcls, dense):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


def _with_int64_indices(mat):
    """The same matrix, forced to carry int64 ``indices``/``indptr``."""
    out = mat.copy()
    out.indices = out.indices.astype(np.int64)
    out.indptr = out.indptr.astype(np.int64)
    return out


# -- the rule itself ---------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, np.int32),
        (1, np.int32),
        (INT32_MAX - 1, np.int32),
        (INT32_MAX, np.int32),
        (INT32_MAX + 1, np.int64),
        (2**40, np.int64),
    ],
)
def test_smallest_index_dtype_boundary(n, expected):
    assert smallest_index_dtype(n) == np.dtype(expected)


# -- construction ------------------------------------------------------------


def test_from_scipy_narrows_int64_indices(dense, vcls):
    """A minor axis that fits int32 is stored as int32, whatever the input carried."""
    mat = _with_int64_indices(_scipy_for(vcls, dense))
    assert mat.indices.dtype == np.int64

    v = vcls.from_scipy(mat)
    assert v.indices.dtype == np.int32
    np.testing.assert_allclose(v.toarray(), dense)


def test_narrowing_halves_the_nnz_sized_array(vcls, rng):
    """``indices`` is the only nnz-sized array, so this is the whole point."""
    dense = rng.integers(0, 4, size=(60, 40)).astype(np.float64)
    mat = _with_int64_indices(_scipy_for(vcls, dense))

    v = vcls.from_scipy(mat)
    assert v.nnz > 0
    assert v.indices.nbytes == 4 * v.nnz
    assert v.indices.nbytes < mat.indices.nbytes


def test_minor_axis_beyond_int32_keeps_int64(vcls):
    """The bound is the axis length, so a genuinely huge axis still gets int64."""
    n_huge = INT32_MAX + 10
    # Two populated major slices against an enormous minor axis: shape is
    # large, nnz is 4, so this stays a tiny allocation.
    minor_idx = np.array([0, INT32_MAX + 5, 1, INT32_MAX + 9], dtype=np.int64)
    shape = (n_huge, 2) if vcls is VCSCArray else (2, n_huge)
    mat_cls = sp.csc_array if vcls is VCSCArray else sp.csr_array
    mat = mat_cls(
        (np.array([1.0, 2.0, 3.0, 4.0]), minor_idx, np.array([0, 2, 4], dtype=np.int64)),
        shape=shape,
    )

    v = vcls.from_scipy(mat)
    assert v.indices.dtype == np.int64
    np.testing.assert_array_equal(np.sort(v.indices), np.sort(minor_idx))


def test_construction_never_widens_narrower_indices(vcls):
    """A caller who already stored something narrower than int32 keeps it."""
    shape = (4, 3) if vcls is VCSCArray else (3, 4)
    v = vcls(
        shape,
        major_ptr=np.array([0, 1, 2, 3], dtype=np.int64),
        values=np.array([1.0, 2.0, 3.0]),
        value_ptr=np.array([0, 1, 2, 3], dtype=np.int64),
        indices=np.array([0, 1, 2], dtype=np.int16),
    )
    assert v.indices.dtype == np.int16


def test_out_of_bounds_index_still_raises_rather_than_truncating(vcls):
    """Narrowing happens after validation, so a bad index is rejected, not wrapped."""
    shape = (4, 1) if vcls is VCSCArray else (1, 4)
    with pytest.raises(ValueError, match="out of bounds"):
        vcls(
            shape,
            major_ptr=np.array([0, 1], dtype=np.int64),
            values=np.array([1.0]),
            value_ptr=np.array([0, 1], dtype=np.int64),
            indices=np.array([2**32 + 1], dtype=np.int64),
        )


def test_transpose_major_narrows_indices(vcls, dense):
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    dual = v._transpose_major()
    assert dual.indices.dtype == np.int32
    np.testing.assert_allclose(dual.toarray(), dense)


# -- write / read round trip -------------------------------------------------


@pytest.mark.parametrize("fmt", ["vcsc", "ivcsc"])
def test_roundtrip_preserves_values_and_stores_narrow_indices(tmp_path, dense, fmt):
    import anndata as ad

    adata = ad.AnnData(X=sp.csr_array(dense))
    va = VCSCAnnData.from_anndata(adata, format="csr")
    path = tmp_path / f"data.{fmt}.h5ad"
    va.write_h5ad(path, format=fmt)

    back = VCSCAnnData.read_h5ad(path)
    assert isinstance(back.X, VCSRArray)
    assert back.X.indices.dtype == np.int32
    np.testing.assert_allclose(back.X.toarray(), dense)
    assert back.X.shape == dense.shape


def test_packed_write_records_narrow_dtype_for_a_wide_in_memory_array(tmp_path, dense):
    """The write path re-derives the dtype rather than trusting the array it's handed."""
    import anndata as ad

    adata = ad.AnnData(X=sp.csr_array(dense))
    va = VCSCAnnData.from_anndata(adata, format="csr")
    assert isinstance(va.X, VCSRArray)
    # Simulate an array built by some other route that kept int64 indices.
    va.X.indices = va.X.indices.astype(np.int64)

    path = tmp_path / "wide.h5ad"
    va.write_h5ad(path, format="ivcsc")

    back = VCSCAnnData.read_h5ad(path)
    assert isinstance(back.X, VCSRArray)
    assert back.X.indices.dtype == np.int32
    np.testing.assert_allclose(back.X.toarray(), dense)


# -- _filter_and_compact: two bounds, two dtypes -----------------------------


def _small_filter_inputs():
    dense = np.array(
        [[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 4.0], [5.0, 0.0, 6.0, 0.0]],
        dtype=np.float32,
    )
    X = sp.csr_array(dense)
    cell_mask = np.array([True, False, True])
    gene_mask = np.array([True, False, True, False])
    return X, cell_mask, gene_mask


def test_filter_and_compact_uses_int32_for_both_when_both_fit():
    X, cell_mask, gene_mask = _small_filter_inputs()
    new_indptr, out_indices, out_data, kept_rows, n_kept = _filter_and_compact(
        X.indptr, X.indices, X.data, cell_mask, gene_mask
    )

    assert new_indptr.dtype == np.int32
    assert out_indices.dtype == np.int32
    assert n_kept == 2
    np.testing.assert_array_equal(kept_rows, [0, 2])
    np.testing.assert_allclose(out_data, [1.0, 2.0, 5.0, 6.0])
    np.testing.assert_array_equal(out_indices, [0, 1, 0, 1])


def test_filter_and_compact_gene_indices_stay_int32_when_pointers_need_int64(monkeypatch):
    """The regression: a big-nnz dataset must not drag the gene indices up with it.

    Allocating a genuinely >INT32_MAX-nonzero matrix isn't testable, so the
    nnz-keyed half of the decision is forced instead -- exactly the situation
    a full-scale dataset produces, where the old shared dtype doubled the
    nnz-sized ``out_indices`` for no reason.
    """
    import vsparse._rapid_load as rapid_load

    X, cell_mask, gene_mask = _small_filter_inputs()
    nnz_in = int(X.indices.shape[0])
    real = rapid_load.smallest_index_dtype

    def forced(n: int) -> np.dtype:
        return np.dtype(np.int64) if n == nnz_in else real(n)

    monkeypatch.setattr(rapid_load, "smallest_index_dtype", forced)
    new_indptr, out_indices, _, _, _ = _filter_and_compact(
        X.indptr, X.indices, X.data, cell_mask, gene_mask
    )

    assert new_indptr.dtype == np.int64  # keyed off nnz: correctly widened
    assert out_indices.dtype == np.int32  # keyed off the gene axis: unaffected
    np.testing.assert_array_equal(out_indices, [0, 1, 0, 1])


# -- the invariant holds across every construction path ----------------------
#
# Narrowing lives in `_VCSBase.__init__` rather than in each method, so any
# operation that builds a new array through `type(self)(...)` inherits it.
# These pin that for the operations added since -- astype, minor-axis
# selection, and the elementwise ops that round-trip through scipy -- so a
# future method can't quietly reintroduce int64 indices for a small axis.


def _int64_indexed(vcls, dense):
    """A VCS array built from an input carrying int64 indices."""
    return vcls.from_scipy(_with_int64_indices(_scipy_for(vcls, dense)))


def test_astype_keeps_narrow_indices(vcls, dense):
    v = _int64_indexed(vcls, dense)
    out = v.astype(np.float32)
    assert out.indices.dtype == np.int32
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.toarray(), dense.astype(np.float32))


def test_minor_axis_selection_keeps_narrow_indices(vcls, dense):
    v = _int64_indexed(vcls, dense)
    n_cols = dense.shape[1]
    cols = np.arange(0, n_cols, 2)

    out = v[:, cols]
    assert out.indices.dtype == np.int32
    np.testing.assert_allclose(out.toarray(), dense[:, cols])


def test_both_axes_selection_keeps_narrow_indices(vcls, dense):
    if dense.shape[0] < 2 or dense.shape[1] < 2:
        pytest.skip("shape too small")
    v = _int64_indexed(vcls, dense)
    rows, cols = np.arange(0, dense.shape[0], 2), np.arange(0, dense.shape[1], 2)

    out = v[rows, :][:, cols]
    assert out.indices.dtype == np.int32
    np.testing.assert_allclose(out.toarray(), dense[np.ix_(rows, cols)])


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(lambda v, d: v * 2.0, id="scalar_mul"),
        pytest.param(lambda v, d: v / 2.0, id="scalar_div"),
        pytest.param(lambda v, d: -v, id="neg"),
        pytest.param(lambda v, d: v + v, id="add_vcs"),
        pytest.param(lambda v, d: v - v, id="sub_vcs"),
        pytest.param(lambda v, d: v.copy(), id="copy"),
        pytest.param(lambda v, d: v.log1p(), id="log1p"),
        pytest.param(lambda v, d: v._transpose_major(), id="transpose_major"),
        pytest.param(lambda v, d: v.T, id="T"),
    ],
)
def test_derived_arrays_keep_narrow_indices(vcls, dense, op):
    """Every op returning a VCS array goes through __init__, so all of them narrow."""
    result = op(_int64_indexed(vcls, dense), dense)
    assert result.indices.dtype == np.int32


def test_scipy_roundtrip_ops_narrow_a_wide_result(vcls, dense):
    """The elementwise ops rebuild via from_scipy, where scipy may hand back int64."""
    v = _int64_indexed(vcls, dense)
    summed = v + v
    assert summed.indices.dtype == np.int32
    np.testing.assert_allclose(summed.toarray(), dense * 2)
