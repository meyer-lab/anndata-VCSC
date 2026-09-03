"""The misaligned-direction matmul regroups a chunk at a time, not the whole array.

The results have to be identical to the full-dual route (covered against a
dense reference in test_vcs_norm.py); what's tested here is that the
chunking machinery itself is correct at boundaries, that multi-chunk runs
actually happen, and that peak memory stays bounded rather than scaling
with the dataset.
"""

from __future__ import annotations

import tracemalloc
from itertools import pairwise

import numpy as np
import pytest
import scipy.sparse as sp

from vsparse import VCSCArray, VCSRArray
from vsparse._vcs_matmul import _TRANSPOSE_BYTES_PER_NNZ, _chunk_bounds


@pytest.fixture(params=[VCSCArray, VCSRArray])
def vcls(request):
    return request.param


def _scipy_for(vcls, dense):
    return sp.csc_array(dense) if vcls is VCSCArray else sp.csr_array(dense)


# -- chunk boundaries --------------------------------------------------------


def test_chunks_cover_the_major_axis_exactly(dense, vcls):
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    for budget in (1, 64, 1 << 20):
        bounds = _chunk_bounds(v, budget)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == v.n_major
        for (_, prev_stop), (start, _) in pairwise(bounds):
            assert start == prev_stop  # contiguous, no gaps or overlaps
        assert all(start < stop for start, stop in bounds)  # always advances


def test_everything_fits_in_one_chunk_under_a_large_budget(dense, vcls):
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    assert _chunk_bounds(v, 1 << 30) == [(0, v.n_major)]


def test_a_tiny_budget_still_makes_progress(vcls):
    """A single major slice bigger than the budget can't be split further."""
    dense = np.ones((40, 40))
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    bounds = _chunk_bounds(v, 1)
    assert bounds == [(i, i + 1) for i in range(v.n_major)]


def test_empty_array_has_no_chunks(vcls):
    shape = (0, 4) if vcls is VCSRArray else (4, 0)
    v = vcls.from_scipy(_scipy_for(vcls, np.zeros(shape)))
    assert _chunk_bounds(v, 1 << 20) == []


def test_chunks_respect_the_budget(vcls, rng):
    dense = rng.integers(0, 4, size=(60, 50)).astype(np.float64)
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    budget = 40 * _TRANSPOSE_BYTES_PER_NNZ  # ~40 nonzeros per chunk

    cumulative = v.value_ptr[v.major_ptr]
    for start, stop in _chunk_bounds(v, budget):
        chunk_nnz = int(cumulative[stop] - cumulative[start])
        # A chunk of one major slice can exceed the budget: it's indivisible.
        assert chunk_nnz <= 40 or stop - start == 1


def test_major_range_is_a_zero_copy_view(vcls, rng):
    """The chunking primitive must not copy the value/index buffers."""
    dense = rng.integers(0, 4, size=(30, 20)).astype(np.float64)
    v = vcls.from_scipy(_scipy_for(vcls, dense))

    chunk = v._major_range(1, 4)
    assert chunk.n_major == 3
    assert np.shares_memory(chunk.values, v.values)
    assert chunk.nnz == 0 or np.shares_memory(chunk.indices, v.indices)


def test_major_range_matches_fancy_selection(vcls, rng):
    dense = rng.integers(0, 4, size=(30, 20)).astype(np.float64)
    v = vcls.from_scipy(_scipy_for(vcls, dense))

    np.testing.assert_allclose(
        v._major_range(2, 7).toarray(),
        v._select_major(np.arange(2, 7)).toarray(),
    )


# -- multi-chunk products match the single-chunk result ----------------------


def _reference(dense: np.ndarray) -> np.ndarray:
    row_totals = dense.sum(axis=1)
    row_scale = row_totals / np.median(row_totals)
    row_scale[row_scale == 0.0] = 1.0
    scaled = dense / row_scale[:, None]
    gene_scale = scaled.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(gene_scale > 0, scaled / gene_scale[None, :], 0.0)
    transformed = np.log10(1.0 + 1000.0 * normalized)
    return transformed - transformed.mean(axis=0, keepdims=True)


@pytest.mark.parametrize("budget", [1, 200, 4000, 1 << 30])
def test_matmul_is_chunk_size_invariant(monkeypatch, vcls, rng, budget):
    """Same answer whether it runs in one chunk or one major slice at a time."""
    import vsparse._vcs_matmul as vcs_matmul

    dense = rng.integers(0, 5, size=(40, 24)).astype(np.float64)
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    ref = _reference(dense)
    B = rng.normal(size=(dense.shape[1], 3))
    Bl = rng.normal(size=(2, dense.shape[0]))

    monkeypatch.setattr(vcs_matmul, "_CHUNK_BUDGET_BYTES", budget)
    nv = v.normalized()

    np.testing.assert_allclose(nv @ B, ref @ B, atol=1e-7)
    np.testing.assert_allclose(Bl @ nv, Bl @ ref, atol=1e-7)
    np.testing.assert_allclose(nv @ B[:, 0], ref @ B[:, 0], atol=1e-7)
    np.testing.assert_allclose(Bl[0] @ nv, Bl[0] @ ref, atol=1e-7)


def test_multi_chunk_matmul_caches_nothing(monkeypatch, vcls, rng):
    """Past one chunk, no full dual is ever built -- that's the memory guarantee."""
    import vsparse._vcs_matmul as vcs_matmul

    dense = rng.integers(1, 5, size=(40, 24)).astype(np.float64)
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    monkeypatch.setattr(vcs_matmul, "_CHUNK_BUDGET_BYTES", 200)
    nv = v.normalized()

    nv @ rng.normal(size=(dense.shape[1], 2))
    rng.normal(size=(2, dense.shape[0])) @ nv

    assert nv._dual_arr is None


def test_many_chunks_actually_run(monkeypatch, vcls, rng):
    """Guards against the budget silently collapsing to a single chunk."""
    import vsparse._vcs_matmul as vcs_matmul

    dense = rng.integers(1, 5, size=(40, 24)).astype(np.float64)
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    monkeypatch.setattr(vcs_matmul, "_CHUNK_BUDGET_BYTES", 200)
    assert len(_chunk_bounds(v, 200)) > 1

    calls = []
    original = vcs_matmul._chunk_bounds
    monkeypatch.setattr(
        vcs_matmul,
        "_chunk_bounds",
        lambda arr, budget: calls.append(len(original(arr, budget))) or original(arr, budget),
    )

    nv = v.normalized()
    nv @ rng.normal(size=(dense.shape[1], 2))
    rng.normal(size=(2, dense.shape[0])) @ nv

    assert calls and max(calls) > 1


# -- the memory bound --------------------------------------------------------


def test_misaligned_matmul_peak_is_bounded_by_the_chunk_budget(monkeypatch, rng):
    """The regression: peak memory tracks the chunk budget, not the dataset.

    A full dual transpose is a second copy of the whole array; this asserts
    the misaligned direction stays far below that even when the budget only
    admits a fraction of the nonzeros at a time.
    """
    import vsparse._vcs_matmul as vcs_matmul

    dense = rng.integers(1, 5, size=(1200, 400)).astype(np.float64)
    v = VCSCArray.from_scipy(sp.csc_array(dense))  # self @ B is misaligned for VCSC
    nnz_bytes = v.nnz * v.indices.dtype.itemsize
    B = rng.normal(size=(dense.shape[1], 2))

    v.normalized() @ B  # warm up numba's JIT before measuring

    monkeypatch.setattr(vcs_matmul, "_CHUNK_BUDGET_BYTES", 64 * _TRANSPOSE_BYTES_PER_NNZ)
    nv = v.normalized()

    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        out = nv @ B
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert nv._dual_arr is None
    assert peak - before < nnz_bytes
    np.testing.assert_allclose(out, _reference(dense) @ B, atol=1e-7)


def test_major_range_agrees_with_native_slicing(vcls, rng):
    """#23 gave __getitem__ a native path for slices; _major_range must match it.

    Both now exist for the same job on a contiguous range -- __getitem__ for
    callers, _major_range as the chunking primitive (it returns views rather
    than gathering). If they ever disagree, the chunked matmul is computing
    against a different sub-array than a caller would get.
    """
    dense = rng.integers(0, 4, size=(30, 20)).astype(np.float64)
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    start, stop = 2, 7

    chunk = v._major_range(start, stop)
    native = v[:, start:stop] if vcls is VCSCArray else v[start:stop, :]

    assert isinstance(native, vcls)
    np.testing.assert_allclose(chunk.toarray(), native.toarray())
