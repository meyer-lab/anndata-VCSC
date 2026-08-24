"""Benchmark VCSCArrayNormalized/VCSRArrayNormalized matmul against scipy and IVCS.

Compares three ways of computing ``nv @ B`` / ``B @ nv`` without ever
materializing the dense normalized matrix:

- ``vcs``: :mod:`vcsc._vcs_matmul` kernels over a plain (already-decoded
  ``indices``) :class:`~vcsc.VCSCArray`/:class:`~vcsc.VCSRArray`.
- ``ivcs``: :mod:`vcsc._ivcs_matmul` kernels, fused with byte-stream
  (varint) decode, over the byte-packed :class:`~vcsc.IVCSCArray`/
  :class:`~vcsc.IVCSRArray`.
- ``scipy``: the natural scipy baseline -- normalize once into a dense
  correction + a scipy ``csr_array``/``csc_array`` of the ``Delta`` term
  (see :mod:`vcsc._vcs_matmul`'s module docstring for the ``A_norm =
  broadcast(-col_mean) + Delta`` decomposition), then let scipy's own
  matmul (single-threaded) do the multiply. Same algebraic decomposition,
  so the comparison isolates kernel/parallelism differences, not different
  math. Note scipy's sparse matmul here is single-threaded, while the vcs/
  ivcs kernels are parallel (numba, all cores) -- so a favorable ratio is
  partly a "more cores" story, not purely an algorithmic one.

Run: ``uv run -p 3.12 python benchmarks/bench_vcs_matmul.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vcsc import IVCSCArray, IVCSRArray, VCSCArray, VCSRArray


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
    fn(*args)  # one warmup call (numba JIT compile / cache load happens here)
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

    vcls = VCSCArray if fmt == "csc" else VCSRArray
    ivcls = IVCSCArray if fmt == "csc" else IVCSRArray
    sp_cls = sp.csc_array if fmt == "csc" else sp.csr_array

    vcs_arr = vcls.from_scipy(sp_cls(dense))
    nv_vcs = vcs_arr.normalized()
    ivcs_arr = ivcls.from_scipy(sp_cls(dense))
    nv_ivcs = ivcs_arr.normalized()

    col_mean, delta_sp = scipy_delta(dense, fmt)
    B = rng.normal(size=(shape[1], k))
    Bl = rng.normal(size=(k, shape[0]))

    def vcs_matmul():
        return nv_vcs @ B

    def ivcs_matmul():
        return nv_ivcs @ B

    def scipy_matmul():
        return delta_sp @ B + (-col_mean) @ B

    def vcs_rmatmul():
        return Bl @ nv_vcs

    def ivcs_rmatmul():
        return Bl @ nv_ivcs

    def scipy_rmatmul():
        return Bl @ delta_sp + np.outer(Bl.sum(axis=1), -col_mean)

    t_vcs = bench(vcs_matmul)
    t_ivcs = bench(ivcs_matmul)
    t_scipy = bench(scipy_matmul)
    t_vcs_r = bench(vcs_rmatmul)
    t_ivcs_r = bench(ivcs_rmatmul)
    t_scipy_r = bench(scipy_rmatmul)

    nnz = int((dense != 0).sum())
    print(
        f"{fmt:>4} shape={shape!s:>14} density={density:<5} k={k:<3} nnz={nnz:<8} | "
        f"matmul  vcs={t_vcs * 1e3:8.3f}ms ivcs={t_ivcs * 1e3:8.3f}ms scipy={t_scipy * 1e3:8.3f}ms "
        f"vcs/scipy={t_vcs / t_scipy:5.2f}x ivcs/scipy={t_ivcs / t_scipy:5.2f}x | "
        f"rmatmul vcs={t_vcs_r * 1e3:8.3f}ms ivcs={t_ivcs_r * 1e3:8.3f}ms scipy={t_scipy_r * 1e3:8.3f}ms "
        f"vcs/scipy={t_vcs_r / t_scipy_r:5.2f}x ivcs/scipy={t_ivcs_r / t_scipy_r:5.2f}x"
    )


if __name__ == "__main__":
    import numba

    print(f"numba threads: {numba.get_num_threads()} (scipy sparse matmul is single-threaded)")
    shapes_and_density = [
        ((2_000, 2_000), 0.20),
        ((10_000, 3_000), 0.05),
        ((30_000, 3_000), 0.02),
    ]
    for shape, density in shapes_and_density:
        for k in (1, 16, 128):
            for fmt in ("csr", "csc"):
                run_case(shape, density, k, fmt)
