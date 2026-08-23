"""Benchmark IVCSCArrayNormalized/IVCSRArrayNormalized matmul against scipy.

Compares ``nv @ B`` / ``B @ nv`` (fused, byte-stream kernels in
:mod:`vcsc._ivcs_matmul`) against the natural scipy baseline: normalize once
into a dense correction + a scipy ``csr_array``/``csc_array`` of the
``Delta`` term (see :mod:`vcsc._ivcs_matmul`'s module docstring for the
``A_norm = broadcast(-col_mean) + Delta`` decomposition), then let scipy's
own (also parallel-BLAS-backed) sparse-dense matmul do the multiply. That is
the fairest baseline: it does the same algebraic decomposition, so the
comparison isolates "fused byte-stream decode + multiply" vs. "materialize
Delta once, then scipy multiplies," not different math.

Run: ``uv run -p 3.12 python benchmarks/bench_matmul.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vcsc import IVCSCArray, IVCSRArray


def make_dense(rng, shape, density):
    dense = rng.integers(0, 8, size=shape).astype(np.float64)
    mask = rng.random(shape) >= density
    dense[mask] = 0.0
    return dense


def scipy_delta(dense: np.ndarray, fmt: str):
    """Build the (dense-correction, scipy-Delta) pair scipy would use."""
    row_totals = dense.sum(axis=1)
    median = np.median(row_totals)
    row_scale = row_totals / median if median > 0 else np.ones_like(row_totals)
    row_scale = np.where(row_scale == 0, 1.0, row_scale)
    scaled = dense / row_scale[:, None]
    gene_scale = scaled.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(gene_scale > 0, scaled / gene_scale[None, :], 0.0)
    transformed = np.log10(1.0 + 1000.0 * normalized)
    col_mean = transformed.mean(axis=0)
    delta = np.where(dense > 0, transformed, 0.0)  # zero off structural nonzeros
    sparse_cls = sp.csc_array if fmt == "csc" else sp.csr_array
    return col_mean, sparse_cls(delta)


def bench(fn, *args, repeats=3):
    # one warmup call (numba JIT compile / cache load happens here)
    fn(*args)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


def run_case(shape, density, k, fmt):
    rng = np.random.default_rng(0)
    dense = make_dense(rng, shape, density)
    if dense.sum() == 0:
        return

    ivcls = IVCSCArray if fmt == "csc" else IVCSRArray
    arr = ivcls.from_scipy((sp.csc_array if fmt == "csc" else sp.csr_array)(dense))
    nv = arr.normalized()

    col_mean, delta_sp = scipy_delta(dense, fmt)
    B = rng.normal(size=(shape[1], k))
    Bl = rng.normal(size=(k, shape[0]))

    def ivcs_matmul():
        return nv @ B

    def scipy_matmul():
        return delta_sp @ B + (-col_mean) @ B

    def ivcs_rmatmul():
        return Bl @ nv

    def scipy_rmatmul():
        return Bl @ delta_sp + np.outer(Bl.sum(axis=1), -col_mean)

    t_ivcs = bench(ivcs_matmul)
    t_scipy = bench(scipy_matmul)
    t_ivcs_r = bench(ivcs_rmatmul)
    t_scipy_r = bench(scipy_rmatmul)

    nnz = int((dense != 0).sum())
    packed_bytes = arr.packed_indices.shape[0]
    print(
        f"{fmt:>4} shape={shape!s:>14} density={density:<5} k={k:<3} nnz={nnz:<8} "
        f"packed={packed_bytes / 1e6:6.2f}MB | "
        f"matmul  ivcs={t_ivcs * 1e3:8.3f}ms scipy={t_scipy * 1e3:8.3f}ms ratio={t_ivcs / t_scipy:5.2f}x | "
        f"rmatmul ivcs={t_ivcs_r * 1e3:8.3f}ms scipy={t_scipy_r * 1e3:8.3f}ms ratio={t_ivcs_r / t_scipy_r:5.2f}x"
    )


if __name__ == "__main__":
    import numba

    print(f"numba threads: {numba.get_num_threads()}")
    shapes_and_density = [
        ((2_000, 2_000), 0.20),
        ((10_000, 3_000), 0.05),
        ((30_000, 3_000), 0.02),
    ]
    for shape, density in shapes_and_density:
        for k in (1, 16, 128):
            for fmt in ("csr", "csc"):
                run_case(shape, density, k, fmt)
