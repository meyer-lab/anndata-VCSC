"""Peak-memory check for the case that regressed in the unbounded-scatter design:
IVCSC.matmul (self @ B) at 30000x3000 density 0.02, k=128 -- see bench_matmul.py.
"""

import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vcsc import IVCSCArray

rng = np.random.default_rng(0)
shape = (30_000, 3_000)
dense = rng.integers(0, 8, size=shape).astype(np.float64)
dense[rng.random(shape) >= 0.02] = 0.0

arr = IVCSCArray.from_scipy(sp.csc_array(dense))
nv = arr.normalized()
B = rng.normal(size=(shape[1], 128))

rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t0 = time.perf_counter()
out = nv @ B  # first call: includes numba JIT compile, but that allocates little
dt = time.perf_counter() - t0
rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(
    f"time={dt * 1e3:.1f}ms out.nbytes={out.nbytes / 1e6:.1f}MB "
    f"maxrss_before={rss_before / 1e3:.1f}MB maxrss_after={rss_after / 1e3:.1f}MB "
    f"delta={(rss_after - rss_before) / 1e3:.1f}MB"
)
