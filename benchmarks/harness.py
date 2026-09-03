"""Measurement primitives for the benchmark suite.

Two kinds of metric, chosen so that a CI job can compare them against a
checked-in threshold without the result depending on which runner it landed
on:

*Memory and layout* numbers are deterministic -- bytes per nonzero is a
property of the data structure, and peak RSS above the data is a property of
what an operation allocates. Both are directly comparable across machines,
which makes them the strongest gates here.

*Timing* numbers are not, so they're never recorded as absolute seconds.
Each timing case measures the same work through scipy in the same process
and reports the **ratio**, which cancels most of the difference between a
fast laptop and a noisy shared CI runner.

Every case runs in its own subprocess (see ``run.py``): ``ru_maxrss`` is a
high-water mark for the life of a process, so cases measured together would
contaminate each other.
"""

from __future__ import annotations

import resource
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse as sp


def peak_rss_mb() -> float:
    """Process peak resident set size, in MB. Monotonic for the process's life."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def peak_alloc_mb(fn: Callable[[], Any]) -> float:
    """Peak memory *allocated during* ``fn``, in MB.

    Not RSS: ``ru_maxrss`` is a high-water mark for the whole process, so an
    operation that stays under the peak set while building its input reports
    zero no matter how much it allocates. ``tracemalloc`` measures the
    allocations themselves and can be reset, which is what makes this
    sensitive enough to gate on.

    numpy allocations are traced; numba's internal (NRT) ones are not. That
    suits the thing being guarded here -- the failure mode is a numpy-level
    temporary the size of the data, not a kernel's own scratch.
    """
    fn()  # JIT compile / warm caches outside the measurement
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        fn()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return max(0.0, (peak - before) / 1e6)


def best_time(fn: Callable[[], Any], repeat: int = 7) -> float:
    """Best wall-clock time over ``repeat`` runs, in seconds. Warms up first."""
    fn()  # JIT compile / allocate caches outside the measurement
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def ratio_vs_scipy(ours: Callable[[], Any], theirs: Callable[[], Any], repeat: int = 7) -> float:
    """``our time / scipy's time`` for the same work. Below 1.0 means we're faster.

    Still the noisiest thing measured here even as a ratio, which is why the
    timing gates carry a much looser margin than the memory ones.
    """
    return best_time(ours, repeat) / best_time(theirs, repeat)


def integer_counts_csr(n_rows: int, n_cols: int, density: float, seed: int = 0) -> sp.csr_array:
    """Integer-valued sparse counts -- the input this package exists for.

    Deliberately integer and heavily repeated: value deduplication is the
    whole premise of the layout, so benchmarking it on unique floats would
    measure a case that never occurs in practice.
    """
    rng = np.random.default_rng(seed)
    mat = sp.random_array((n_rows, n_cols), density=density, format="csr", random_state=seed)
    mat.data = np.round(rng.integers(1, 8, size=mat.data.shape[0])).astype(np.float64)
    return mat
