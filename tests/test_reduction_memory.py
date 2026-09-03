"""Minor-axis reductions: correctness, and the memory bound that makes them usable.

Every reduction along the minor axis is a scatter over each stored nonzero.
Written the obvious way -- expand the value-compressed layout to one value
per nonzero, then scatter with ``np.repeat``/``ufunc.at``/``np.bincount`` --
each one allocates an nnz-sized temporary as scratch for a result of length
``n_minor``. This module covers the kernels that avoid that, and pins the
bound so the pattern can't come back.

Reduction *semantics* (implicit-zero handling, axis conventions, arithmetic)
live in test_reductions_and_arith.py; this file is about the cost.
"""

from __future__ import annotations

import tracemalloc

import numba
import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCArray, VCSRArray
from vsparse._ops import (
    _ACCUMULATOR_BUDGET_BYTES,
    _accumulator_threads,
    minor_counts,
    minor_extrema,
    minor_sums,
)


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def _scipy_for(vcls, dense):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


# -- correctness against a dense reference -----------------------------------


def test_sum_all_matches_dense(dense, vcls):
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    assert v.sum() == pytest.approx(float(dense.sum()))


@pytest.mark.parametrize("axis", [0, 1])
def test_sum_axis_matches_dense(dense, vcls, axis):
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    np.testing.assert_allclose(v.sum(axis=axis), dense.sum(axis=axis))


def test_sum_bad_axis_raises(dense, vcls):
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    with pytest.raises(ValueError, match="axis must be"):
        v.sum(axis=2)


def test_minor_sums_matches_expanded_reference(dense, vcls):
    """Explicitly against the expand-then-bincount formula this replaces."""
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    expanded = np.repeat(v.values.astype(np.float64), np.diff(v.value_ptr))
    reference = np.bincount(v.indices, weights=expanded, minlength=v.n_minor)
    np.testing.assert_allclose(v._minor_sums(), reference)


def test_minor_sums_on_empty_array(vcls):
    dense = np.zeros((6, 5))
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    np.testing.assert_allclose(v._minor_sums(), np.zeros(v.n_minor))


def test_minor_sums_returns_float64_for_integer_values(vcls):
    """Accumulation is float64 regardless of the stored value dtype."""
    dense = np.array([[1, 0, 2], [3, 4, 0]], dtype=np.int32)
    v = vcls.from_scipy(_scipy_for(vcls, dense.astype(np.float64)))
    assert v._minor_sums().dtype == np.float64


def test_minor_sums_direct_call(vcls):
    dense = np.array([[1.0, 0.0, 2.0], [3.0, 4.0, 0.0]])
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    out = minor_sums(v.values, v.value_ptr, v.indices, v.n_minor)
    np.testing.assert_allclose(out, dense.sum(axis=0 if vcls is VCSRArray else 1))


# -- the accumulator budget --------------------------------------------------


@pytest.mark.parametrize("n_minor", [0, 1, 1_000, 33_538, 10**6, 10**8])
@pytest.mark.parametrize("bytes_per_element", [8, 16])
def test_accumulator_block_stays_within_budget(n_minor, bytes_per_element):
    """Thread-local accumulators must not become the new unbounded allocation."""
    nthreads = _accumulator_threads(n_minor, bytes_per_element)
    assert nthreads >= 1
    assert nthreads <= numba.get_num_threads()
    if nthreads > 1:
        assert nthreads * n_minor * bytes_per_element <= _ACCUMULATOR_BUDGET_BYTES


def test_wider_accumulators_get_fewer_threads():
    """A kernel keeping more per-slot state must not blow the same budget."""
    n_minor = _ACCUMULATOR_BUDGET_BYTES // (8 * 4)  # 4 threads' worth at 8 bytes
    assert _accumulator_threads(n_minor, 16) <= _accumulator_threads(n_minor, 8)


def test_wide_minor_axis_falls_back_to_one_thread():
    """A minor axis too wide to afford even two accumulators runs serially."""
    assert _accumulator_threads(_ACCUMULATOR_BUDGET_BYTES) == 1


def test_narrow_minor_axis_uses_all_threads():
    assert _accumulator_threads(64) == numba.get_num_threads()


# -- the memory bound this replaces ------------------------------------------


def test_minor_sums_allocates_nothing_nnz_sized():
    """The regression guard: no nnz-sized temporary anywhere in the call path.

    The implementation this replaced expanded the value-compressed layout
    back to one float64 per nonzero (``np.repeat``) purely as scratch for
    the reduction -- 8 bytes per nonzero, which at cohort scale is tens of
    GiB. Sized so that the old temporary would be ~8 MiB while the bound
    asserted here is 1 MiB.
    """
    rng = np.random.default_rng(0)
    n_rows, n_cols = 2_000, 500
    dense = rng.integers(1, 5, size=(n_rows, n_cols)).astype(np.float64)
    v = VCSRArray.from_scipy(sp.csr_array(dense))
    assert v.nnz == n_rows * n_cols  # 1e6 nonzeros: old scratch would be 8 MB

    v._minor_sums()  # warm up numba's JIT before measuring

    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        out = v._minor_sums()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    allocated = peak - before
    assert allocated < 1 << 20, f"allocated {allocated / 1e6:.1f} MB for {v.nnz} nonzeros"
    np.testing.assert_allclose(out, dense.sum(axis=0))


# -- max / min / getnnz: the same defect, in the operations added later ------
#
# #22 added per-axis max/min and getnnz, each written with the expand-then-
# scatter pattern (np.repeat + ufunc.at, and np.bincount, which promotes an
# int32 `indices` to intp before counting). These pin the replacements: same
# results, without the nnz-sized temporary.


def _old_minor_reduce(v, ufunc, initial):
    """The expand-then-scatter implementation these kernels replaced."""
    expanded = np.repeat(v.values, np.diff(v.value_ptr))
    out = np.full(v.n_minor, initial, dtype=v.values.dtype)
    ufunc.at(out, v.indices, expanded)
    counts = np.bincount(v.indices, minlength=v.n_minor)
    not_dense = counts < v.n_major
    out[not_dense] = ufunc(out[not_dense], 0)
    return out


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("kind", ["max", "min"])
def test_extrema_match_dense(dense, vcls, axis, kind):
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    expected = getattr(dense, kind)(axis=axis)
    np.testing.assert_allclose(getattr(v, kind)(axis=axis), expected)


@pytest.mark.parametrize("kind", ["max", "min"])
def test_minor_extrema_match_the_expanded_reference(dense, vcls, kind):
    """Explicitly against the implementation being replaced, not just a dense truth."""
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    ufunc = np.maximum if kind == "max" else np.minimum
    initial = -np.inf if kind == "max" else np.inf
    minor_axis = 1 if vcls is VCSCArray else 0
    np.testing.assert_allclose(
        getattr(v, kind)(axis=minor_axis), _old_minor_reduce(v, ufunc, initial)
    )


def test_minor_nnz_matches_bincount_reference(dense, vcls):
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    reference = np.bincount(v.indices, minlength=v.n_minor).astype(np.int64)
    np.testing.assert_array_equal(v._minor_nnz(), reference)
    np.testing.assert_array_equal(
        v._minor_nnz(), (dense != 0).sum(axis=1 if vcls is VCSCArray else 0)
    )


def test_extrema_kernel_reports_stored_extremum_and_count(vcls):
    """The kernel deliberately ignores implicit zeros; the count is how callers find them."""
    dense = np.array([[3.0, 0.0], [5.0, 0.0]])
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    extrema, counts = minor_extrema(
        v.values, v.value_ptr, v.indices, v.n_minor, -np.inf, True
    )
    assert extrema.shape == (v.n_minor,)
    assert counts.sum() == v.nnz
    np.testing.assert_array_equal(counts, minor_counts(v.value_ptr, v.indices, v.n_minor))


def test_extrema_on_empty_array(vcls):
    """No stored values at all: every entry is an implicit zero."""
    v = vcls.from_scipy(_scipy_for(vcls, np.zeros((4, 3))))
    np.testing.assert_allclose(v.max(axis=0), np.zeros(3))
    np.testing.assert_allclose(v.min(axis=0), np.zeros(3))
    np.testing.assert_array_equal(v.getnnz(axis=0), np.zeros(3, dtype=np.int64))


def test_integer_dtype_extrema_use_integer_sentinels(vcls):
    """The identity element comes from the stored dtype, so integers stay exact."""
    dense = np.array([[7, 0, 2], [0, 3, 9]], dtype=np.int32)
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    np.testing.assert_array_equal(v.max(axis=0), dense.max(axis=0))
    np.testing.assert_array_equal(v.min(axis=0), dense.min(axis=0))


@pytest.mark.parametrize(
    ("label", "call"),
    [
        ("max", lambda v: v.max(axis=0)),
        ("min", lambda v: v.min(axis=0)),
        ("getnnz", lambda v: v.getnnz(axis=0)),
    ],
)
def test_minor_axis_ops_allocate_nothing_nnz_sized(label, call):
    """Same bound as the sum: an n_minor-sized result must not cost nnz-sized scratch."""
    rng = np.random.default_rng(0)
    n_rows, n_cols = 2_000, 500
    dense = rng.integers(1, 5, size=(n_rows, n_cols)).astype(np.float64)
    v = VCSRArray.from_scipy(sp.csr_array(dense))
    assert v.nnz == n_rows * n_cols  # 1e6 nonzeros: old scratch was 8-16 MB

    call(v)  # warm up numba's JIT before measuring

    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        out = call(v)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    allocated = peak - before
    assert allocated < 1 << 20, f"{label} allocated {allocated / 1e6:.1f} MB for {v.nnz} nonzeros"
    assert out.shape == (n_cols,)
