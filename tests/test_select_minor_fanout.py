"""Minor-axis selection is a fan-out: one stored element can land in many outputs.

``arr[:, [1, 1, 3]]`` names column 1 twice. scipy supports that, and so did
this package until minor-axis selection got a native path -- at which point
a single old->new lookup array collapsed every repeat but the last and
returned zeros in the dropped slots, with no error. These tests pin the
fan-out semantics against scipy, which is the reference the native path has
to reproduce.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCArray, VCSRArray


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def _scipy_for(vcls, dense):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


def _grid(n_rows: int = 4, n_cols: int = 6) -> np.ndarray:
    """Distinct values everywhere, so any misplacement is visible."""
    return np.arange(1, n_rows * n_cols + 1, dtype=float).reshape(n_rows, n_cols)


# -- the regression ----------------------------------------------------------


@pytest.mark.parametrize(
    "cols",
    [
        pytest.param([1, 1, 3], id="one_repeat"),
        pytest.param([2, 2, 2], id="same_index_three_times"),
        pytest.param([0, 0, 0, 0], id="one_index_only"),
        pytest.param([3, 1, 3, 1], id="interleaved_repeats"),
        pytest.param([0, 1, 1, 2, 2, 2], id="mixed_multiplicities"),
    ],
)
def test_duplicate_minor_indices_fan_out(vcls, cols):
    dense = _grid()
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    np.testing.assert_allclose(v[:, cols].toarray(), dense[:, cols])


def test_duplicate_indices_on_the_other_axis_too(vcls):
    """Whichever axis is the *minor* one for this format takes the same path."""
    dense = _grid()
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    rows = [1, 1, 2, 0, 0]
    np.testing.assert_allclose(v[rows, :].toarray(), dense[rows, :])


def test_duplicates_on_both_axes_at_once(vcls):
    dense = _grid()
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    rows, cols = [0, 0, 2], [1, 1, 4]
    np.testing.assert_allclose(v[rows, cols].toarray(), dense[np.ix_(rows, cols)])


def test_matches_scipy_for_a_random_selection_with_repeats(vcls, rng):
    """Against scipy directly -- the behaviour that regressed."""
    dense = rng.integers(0, 4, size=(12, 9)).astype(np.float64)
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    reference = _scipy_for(vcls, dense)

    for _ in range(25):
        cols = rng.integers(0, dense.shape[1], size=rng.integers(1, 15)).tolist()
        np.testing.assert_allclose(
            v[:, cols].toarray(), np.asarray(reference[:, cols].todense())
        )


# -- selections that already worked, kept working ----------------------------


@pytest.mark.parametrize(
    "cols",
    [
        pytest.param([0, 2], id="sorted"),
        pytest.param([2, 0], id="reversed"),
        pytest.param(np.array([], dtype=int), id="empty"),
        pytest.param([-1, -2], id="negative"),
        pytest.param(slice(1, 4), id="slice"),
        pytest.param(slice(None, None, 2), id="strided_slice"),
        pytest.param([True, False, True, False, True, False], id="boolean_mask"),
    ],
)
def test_non_duplicate_selections_unchanged(vcls, cols):
    dense = _grid()
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    expected = dense[:, cols] if isinstance(cols, slice) else dense[:, np.asarray(cols)]
    np.testing.assert_allclose(v[:, cols].toarray(), expected)


def test_selection_keeps_the_vcs_type_and_index_dtype(vcls):
    dense = _grid()
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    out = v[:, [1, 1, 3]]
    assert isinstance(out, vcls)
    assert out.indices.dtype == v.indices.dtype
    assert out.shape == (dense.shape[0], 3) if vcls is VCSRArray else out.shape == (4, 3)


def test_empty_selection_gives_a_zero_width_array(vcls):
    dense = _grid()
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    out = v[:, []]
    assert out.shape[1] == 0
    assert out.nnz == 0


def test_selection_of_an_all_zero_array(vcls):
    v = vcls.from_scipy(_scipy_for(vcls, np.zeros((4, 5))))
    out = v[:, [1, 1, 2]]
    assert out.nnz == 0
    np.testing.assert_allclose(out.toarray(), np.zeros((4, 3)))


def test_sparse_columns_and_duplicates_together(vcls):
    """A duplicated index whose column is entirely implicit zeros."""
    dense = np.array([[1.0, 0.0, 2.0], [3.0, 0.0, 0.0]])
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    cols = [1, 1, 0, 2]
    np.testing.assert_allclose(v[:, cols].toarray(), dense[:, cols])


def test_round_trip_through_scipy_after_a_duplicate_selection(vcls):
    """The result has to be a structurally valid VCS array, not just print right."""
    dense = _grid()
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    out = v[:, [1, 1, 3]]

    rebuilt = vcls.from_scipy(out.to_scipy())
    np.testing.assert_allclose(rebuilt.toarray(), dense[:, [1, 1, 3]])
    assert out.value_ptr[-1] == out.indices.shape[0]
    assert out.major_ptr[-1] == out.values.shape[0]


# -- memory ------------------------------------------------------------------


def test_minor_selection_allocates_nothing_nnz_sized():
    """The rewrite also drops the nnz-sized intermediates the remap version needed.

    Sized so the old implementation's per-nonzero temporaries were ~30 MB
    while the bound here is 16 MB -- the result itself is legitimately
    nnz-scale, so this bounds the *scratch*, not the output.
    """
    rng = np.random.default_rng(0)
    n_rows, n_cols = 2_000, 500
    dense = rng.integers(1, 5, size=(n_rows, n_cols)).astype(np.float64)
    v = VCSRArray.from_scipy(sp.csr_array(dense))
    cols = np.arange(0, n_cols, 2)

    v[:, cols]  # warm up numba's JIT before measuring

    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        out = v[:, cols]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak - before < 16 << 20, f"allocated {(peak - before) / 1e6:.1f} MB"
    np.testing.assert_allclose(out.toarray(), dense[:, cols])
