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


@pytest.mark.parametrize("budget", [1, 200, 1 << 30])
def test_chunks_tile_the_major_axis(dense, vcls, budget):
    """Chunks must cover every major slice exactly once, at any budget."""
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    bounds = _chunk_bounds(v, budget)

    assert bounds[0][0] == 0
    assert bounds[-1][1] == v.n_major
    assert all(start < stop for start, stop in bounds)
    for (_, prev_stop), (start, _) in pairwise(bounds):
        assert start == prev_stop


def test_chunks_respect_the_budget(vcls, rng):
    """A chunk may only exceed the budget when it is a single, indivisible slice."""
    dense = rng.integers(0, 4, size=(60, 50)).astype(np.float64)
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    budget = 40 * _TRANSPOSE_BYTES_PER_NNZ

    cumulative = v.value_ptr[v.major_ptr]
    for start, stop in _chunk_bounds(v, budget):
        chunk_nnz = int(cumulative[stop] - cumulative[start])
        assert chunk_nnz <= 40 or stop - start == 1


def test_empty_array_has_no_chunks(vcls):
    shape = (0, 4) if vcls is VCSRArray else (4, 0)
    v = vcls.from_scipy(_scipy_for(vcls, np.zeros(shape)))
    assert _chunk_bounds(v, 1 << 20) == []


def test_major_range_is_a_zero_copy_view(vcls, rng):
    """Chunking is only affordable because the buffers are shared, not gathered."""
    dense = rng.integers(0, 4, size=(30, 20)).astype(np.float64)
    v = vcls.from_scipy(_scipy_for(vcls, dense))

    chunk = v._major_range(1, 4)
    assert np.shares_memory(chunk.values, v.values)
    assert chunk.nnz == 0 or np.shares_memory(chunk.indices, v.indices)


def test_major_range_matches_ordinary_slicing(vcls, rng):
    """A chunk must hold the same sub-array a caller would get by slicing."""
    dense = rng.integers(0, 4, size=(30, 20)).astype(np.float64)
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    start, stop = 2, 7

    native = v[:, start:stop] if vcls is VCSCArray else v[start:stop, :]
    np.testing.assert_allclose(v._major_range(start, stop).toarray(), native.toarray())


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
    """Past one chunk no full dual is built, which is the memory guarantee."""
    import vsparse._vcs_matmul as vcs_matmul

    dense = rng.integers(1, 5, size=(40, 24)).astype(np.float64)
    v = vcls.from_scipy(_scipy_for(vcls, dense))
    monkeypatch.setattr(vcs_matmul, "_CHUNK_BUDGET_BYTES", 200)
    assert len(_chunk_bounds(v, 200)) > 1

    nv = v.normalized()
    nv @ rng.normal(size=(dense.shape[1], 2))
    rng.normal(size=(2, dense.shape[0])) @ nv

    assert nv._dual_arr is None


def test_misaligned_matmul_peak_is_bounded_by_the_chunk_budget(monkeypatch, rng):
    """Peak memory has to track the budget, not the size of the array."""
    import vsparse._vcs_matmul as vcs_matmul

    dense = rng.integers(1, 5, size=(1200, 400)).astype(np.float64)
    v = VCSCArray.from_scipy(sp.csc_array(dense))
    nnz_bytes = v.nnz * v.indices.dtype.itemsize
    B = rng.normal(size=(dense.shape[1], 2))

    v.normalized() @ B  # warm up the JIT before measuring

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
