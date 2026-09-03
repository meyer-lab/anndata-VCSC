"""Windowing a normalized view vs. renormalizing a selection -- and why they differ.

A normalized view's statistics are fixed at construction, over the whole
array it wrapped. Indexing it is a window into *that* matrix; ``select``
renormalizes the chosen sub-array on its own terms. Getting those two
confused silently fits downstream analysis to the wrong data, so both
contracts are pinned here, along with the size of the gap between them.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCArray, VCSCArrayNormalized, VCSRArray, VCSRArrayNormalized


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def _scipy_for(vcls, dense):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


def _norm_cls(vcls):
    return VCSCArrayNormalized if vcls is VCSCArray else VCSRArrayNormalized


def _reference(dense: np.ndarray) -> np.ndarray:
    """Read-depth normalize, log-transform, and mean-center a dense matrix directly."""
    row_totals = dense.sum(axis=1)
    row_scale = row_totals / np.median(row_totals)
    row_scale[row_scale == 0.0] = 1.0
    scaled = dense / row_scale[:, None]
    gene_scale = scaled.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(gene_scale > 0, scaled / gene_scale[None, :], 0.0)
    transformed = np.log10(1.0 + 1000.0 * normalized)
    return transformed - transformed.mean(axis=0, keepdims=True)


def _mixed_population(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two cell types: different sequencing depth, different marker genes.

    This is the shape of data the distinction actually matters for -- a
    selection whose depth and expression profile both differ from the
    population the view was built over.
    """
    rng = np.random.default_rng(seed)
    n_cells, n_genes = 240, 80
    dense = np.empty((n_cells, n_genes))
    dense[: n_cells // 2] = rng.poisson(0.6, size=(n_cells // 2, n_genes))
    dense[n_cells // 2 :] = rng.poisson(4.0, size=(n_cells // 2, n_genes))
    dense[n_cells // 2 :, :16] *= 6  # markers expressed only in the second type

    mask = np.zeros(n_cells, dtype=bool)
    mask[n_cells // 2 :] = True
    return dense, mask


def _relative_frobenius(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


# -- select(): statistics recomputed for the selection -----------------------


def test_select_matches_normalizing_the_selection_directly(vcls):
    """The correctness claim: select() == normalizing those cells on their own."""
    dense, mask = _mixed_population()
    nv = vcls.from_scipy(_scipy_for(vcls, dense)).normalized()

    got = nv.select(mask).toarray()
    want = _reference(dense[mask])

    assert _relative_frobenius(got, want) < 1e-12
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_select_equals_selecting_on_the_raw_array_first(vcls):
    """select() is exactly the arr[sel].normalized() route, reachable from the view.

    Since #23 the raw route stays VCS-native on both axes, so this is now a
    direct comparison; ``select`` remains the discoverable spelling, and the
    one that doesn't require reaching for the view's private ``_arr``.
    """
    dense, mask = _mixed_population()
    arr = vcls.from_scipy(_scipy_for(vcls, dense))

    raw_sub = arr[mask, :]
    assert isinstance(raw_sub, vcls)

    np.testing.assert_allclose(
        arr.normalized().select(mask).toarray(),
        raw_sub.normalized().toarray(),
        atol=1e-12,
    )


def test_select_returns_a_view_that_still_composes(vcls):
    """Not a dense block: it keeps working with @ and toarray()."""
    dense, mask = _mixed_population()
    nv = vcls.from_scipy(_scipy_for(vcls, dense)).normalized()

    sub = nv.select(mask)
    assert isinstance(sub, _norm_cls(vcls))
    assert sub.shape == (int(mask.sum()), dense.shape[1])

    rng = np.random.default_rng(3)
    B = rng.normal(size=(dense.shape[1], 4))
    np.testing.assert_allclose(sub @ B, _reference(dense[mask]) @ B, atol=1e-8)


def test_select_columns_and_both_axes(vcls):
    dense, _ = _mixed_population()
    nv = vcls.from_scipy(_scipy_for(vcls, dense)).normalized()
    cols = np.arange(0, dense.shape[1], 3)

    np.testing.assert_allclose(
        nv.select(cols=cols).toarray(), _reference(dense[:, cols]), atol=1e-10
    )

    rows = np.arange(0, dense.shape[0], 5)
    np.testing.assert_allclose(
        nv.select(rows, cols).toarray(), _reference(dense[np.ix_(rows, cols)]), atol=1e-10
    )


def test_select_everything_is_the_whole_view(vcls, dense):
    if dense.sum() == 0:
        pytest.skip("all-zero matrix: median row total is 0")
    nv = vcls.from_scipy(_scipy_for(vcls, dense)).normalized()
    np.testing.assert_allclose(nv.select().toarray(), nv.toarray(), atol=1e-12)


# -- __getitem__: a window that keeps the parent's statistics ----------------


def test_getitem_is_a_window_into_the_parent_matrix(vcls):
    """Indexing == toarray()[key], without materializing the full matrix."""
    dense, mask = _mixed_population()
    nv = vcls.from_scipy(_scipy_for(vcls, dense)).normalized()

    full = nv.toarray()
    np.testing.assert_allclose(nv[mask, :], full[mask, :], atol=1e-10)
    np.testing.assert_allclose(nv[0:4, 0:4], full[0:4, 0:4], atol=1e-10)


def test_getitem_and_select_disagree_substantially_on_a_real_selection(vcls):
    """The hazard, made explicit and measurable.

    Reading ``view[cell_type]`` as "the normalized data for these cells" is
    wrong by tens of percent -- large enough that any factorization fit to
    it is fitting different data than the caller thinks. This test exists so
    that the difference can't be quietly collapsed by a future change: if
    these two ever agree, one of the two contracts has been broken.
    """
    dense, mask = _mixed_population()
    nv = vcls.from_scipy(_scipy_for(vcls, dense)).normalized()

    want = _reference(dense[mask])
    windowed = np.asarray(nv[mask, :])
    renormalized = nv.select(mask).toarray()

    assert _relative_frobenius(renormalized, want) < 1e-12
    assert _relative_frobenius(windowed, want) > 0.1


def test_select_survives_native_two_axis_indexing(vcls):
    """#23 made both-axes selection VCS-native; select() must still get outer semantics.

    A single ``arr[rows, cols]`` used to fall through to scipy, which
    broadcasts two index arrays against each other pointwise. It now
    composes a major- and a minor-axis selection instead, which is the
    sub-block ``select`` needs -- worth pinning, since the difference is
    silent and only shows up when both axes are arrays.
    """
    dense, _ = _mixed_population()
    nv = vcls.from_scipy(_scipy_for(vcls, dense)).normalized()
    rows = np.arange(0, dense.shape[0], 7)
    cols = np.arange(0, dense.shape[1], 5)
    assert rows.shape != cols.shape  # a pointwise broadcast would fail outright

    np.testing.assert_allclose(
        nv.select(rows, cols).toarray(), _reference(dense[np.ix_(rows, cols)]), atol=1e-10
    )
