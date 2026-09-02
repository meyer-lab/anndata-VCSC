"""Tests for VCSC/VCSR axis reductions, and the memory bound on the minor-axis one."""

from __future__ import annotations

import tracemalloc

import numba
import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCArray, VCSRArray
from vsparse._ops import _ACCUMULATOR_BUDGET_BYTES, _accumulator_threads, minor_sums


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
def test_accumulator_block_stays_within_budget(n_minor):
    """Thread-local accumulators must not become the new unbounded allocation."""
    nthreads = _accumulator_threads(n_minor)
    assert nthreads >= 1
    assert nthreads <= numba.get_num_threads()
    if nthreads > 1:
        assert nthreads * n_minor * 8 <= _ACCUMULATOR_BUDGET_BYTES


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
