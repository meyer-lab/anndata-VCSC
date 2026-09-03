"""The benchmark cases themselves.

Each case is a function returning ``{metric_name: value}``. Cases marked
``fast`` run on every pull request; the rest are for the scheduled job and
for running by hand on real data.

What's here is chosen to cover the claims that are easy to regress silently:
the size of the stored layout, and whether an operation allocates a second
copy of the data. A correctness test won't catch either -- the answers stay
right while the cost quietly doubles.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from benchmarks.harness import (
    integer_counts_csr,
    peak_alloc_mb,
    ratio_vs_scipy,
)

FAST: dict[str, Callable[[], dict[str, float]]] = {}
SLOW: dict[str, Callable[[], dict[str, float]]] = {}


def fast(fn):
    FAST[fn.__name__] = fn
    return fn


def slow(fn):
    SLOW[fn.__name__] = fn
    return fn


# -- layout size -------------------------------------------------------------


@fast
def layout_bytes_per_nonzero() -> dict[str, float]:
    """Total bytes the VCS layout holds, per stored nonzero.

    Deterministic, so this is an exact gate rather than a noisy one. It
    moves if index dtypes widen, if deduplication stops working, or if a
    new array joins the layout.
    """
    from vsparse import VCSRArray

    mat = integer_counts_csr(20_000, 2_000, density=0.05)
    v = VCSRArray.from_scipy(mat)
    stored = v.major_ptr.nbytes + v.values.nbytes + v.value_ptr.nbytes + v.indices.nbytes
    scipy_bytes = mat.indptr.nbytes + mat.indices.nbytes + mat.data.nbytes
    return {
        "bytes_per_nonzero": stored / v.nnz,
        "indices_bytes_per_nonzero": v.indices.nbytes / v.nnz,
        "vs_scipy_ratio": stored / scipy_bytes,
    }


# -- memory ceilings ---------------------------------------------------------


@fast
def minor_sum_peak_mb() -> dict[str, float]:
    """Memory allocated by a minor-axis reduction.

    A reduction produces an ``n_minor``-sized result; anything approaching
    the size of the data means an nnz-sized temporary crept back in.
    """
    from vsparse import VCSRArray

    v = VCSRArray.from_scipy(integer_counts_csr(40_000, 2_000, density=0.05))
    nnz_mb = v.nnz * 8 / 1e6
    return {
        "peak_alloc_mb": peak_alloc_mb(lambda: v.sum(axis=0)),
        "expanded_nnz_mb": nnz_mb,  # what a per-nonzero temporary would cost
    }


@fast
def misaligned_matmul_peak_mb() -> dict[str, float]:
    """Memory allocated by the matmul direction the storage isn't aligned for.

    The failure mode is a full opposite-format copy of the array, so this
    should stay far below the array's own footprint.
    """
    from vsparse import VCSCArray

    v = VCSCArray.from_scipy(integer_counts_csr(40_000, 2_000, density=0.05))
    rng = np.random.default_rng(0)
    B = rng.normal(size=(v.shape[1], 4))
    array_mb = (v.values.nbytes + v.value_ptr.nbytes + v.indices.nbytes) / 1e6

    return {
        "peak_alloc_mb": peak_alloc_mb(lambda: v.normalized() @ B),
        "array_mb": array_mb,  # what a full second copy would cost
    }


@fast
def minor_extrema_peak_mb() -> dict[str, float]:
    """Memory allocated by a minor-axis max/min.

    Added because #22's `max`/`min` shipped with the same expand-then-scatter
    temporary the reduction case above exists to catch, which is the clearest
    evidence available that this gate is worth having: the pattern reappears
    in new code unless something measures it.
    """
    from vsparse import VCSRArray

    v = VCSRArray.from_scipy(integer_counts_csr(40_000, 2_000, density=0.05))
    return {
        "peak_alloc_mb": peak_alloc_mb(lambda: v.max(axis=0)),
        "expanded_nnz_mb": v.nnz * 8 / 1e6,
    }


@fast
def minor_getnnz_peak_mb() -> dict[str, float]:
    """Memory allocated by a per-minor-index stored-element count.

    np.bincount is the obvious implementation and promotes an int32
    ``indices`` to intp first, so this is nnz-sized for an n_minor-sized
    answer -- invisible in a correctness test.
    """
    from vsparse import VCSRArray

    v = VCSRArray.from_scipy(integer_counts_csr(40_000, 2_000, density=0.05))
    return {
        "peak_alloc_mb": peak_alloc_mb(lambda: v.getnnz(axis=0)),
        "indices_nnz_mb": v.nnz * 8 / 1e6,
    }


@fast
def minor_selection_peak_mb() -> dict[str, float]:
    """Memory allocated by a minor-axis selection (#23's `_select_minor`).

    Currently carries an nnz-sized temporary of its own (ISSUE-30); recorded
    so the number is tracked rather than assumed, and so the ceiling drops
    visibly when that is fixed.
    """
    from vsparse import VCSRArray

    v = VCSRArray.from_scipy(integer_counts_csr(40_000, 2_000, density=0.05))
    cols = np.arange(0, v.shape[1], 2)
    return {
        "peak_alloc_mb": peak_alloc_mb(lambda: v[:, cols]),
        "indices_nnz_mb": v.nnz * 8 / 1e6,
    }


# -- throughput, relative to scipy -------------------------------------------


@fast
def matvec_vs_scipy() -> dict[str, float]:
    """Aligned-direction matrix-vector product, against scipy's CSR."""
    from vsparse import VCSRArray

    mat = integer_counts_csr(20_000, 2_000, density=0.05)
    v = VCSRArray.from_scipy(mat)
    rng = np.random.default_rng(0)
    x = rng.normal(size=mat.shape[1])
    return {"time_ratio_vs_scipy": ratio_vs_scipy(lambda: v @ x, lambda: mat @ x)}


@fast
def matmat_vs_scipy() -> dict[str, float]:
    """Aligned-direction matrix-matrix product, against scipy's CSR."""
    from vsparse import VCSRArray

    mat = integer_counts_csr(20_000, 2_000, density=0.05)
    v = VCSRArray.from_scipy(mat)
    rng = np.random.default_rng(0)
    B = rng.normal(size=(mat.shape[1], 8))
    return {"time_ratio_vs_scipy": ratio_vs_scipy(lambda: v @ B, lambda: mat @ B)}


# -- larger, for the scheduled job -------------------------------------------


@slow
def large_layout_and_matmul() -> dict[str, float]:
    """The same shape of measurement an order of magnitude up."""
    from vsparse import VCSRArray

    mat = integer_counts_csr(200_000, 3_000, density=0.02)
    v = VCSRArray.from_scipy(mat)
    rng = np.random.default_rng(0)
    B = rng.normal(size=(mat.shape[1], 8))
    stored = v.major_ptr.nbytes + v.values.nbytes + v.value_ptr.nbytes + v.indices.nbytes
    v @ B
    return {
        "bytes_per_nonzero": stored / v.nnz,
        "time_ratio_vs_scipy": ratio_vs_scipy(lambda: v @ B, lambda: mat @ B),
        "matmul_peak_alloc_mb": peak_alloc_mb(lambda: v @ B),
    }


ALL: dict[str, Callable[[], dict[str, float]]] = {**FAST, **SLOW}
