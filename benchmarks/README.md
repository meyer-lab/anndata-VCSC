# Benchmarks

A small suite that CI runs as a regression gate, plus a larger set for
running by hand or on a schedule.

```sh
uv run python -m benchmarks.run --set fast       # run + compare to baselines (what CI does)
uv run python -m benchmarks.run --set slow       # the larger cases
uv run python -m benchmarks.run --case matvec_vs_scipy
uv run python -m benchmarks.run --set fast --record   # rewrite baselines.json
```

Exits nonzero if any gated metric exceeds its recorded ceiling.

## What it measures, and why those things

Correctness tests don't catch cost regressions: the answers stay right while
the memory or the file doubles. These are the two properties that can move
silently.

**Layout size** — bytes per stored nonzero, and the same figure for the
`indices` array alone. Deterministic, so it's an exact gate. It moves if an
index dtype widens, if value deduplication stops working, or if a new array
joins the layout.

**Memory allocated by an operation** — measured with `tracemalloc`, not
`ru_maxrss`. RSS is a process-lifetime high-water mark, so an operation that
stays under the peak set while building its input reports zero no matter how
much it allocates; that made the first version of this suite report `0` for
an operation allocating 66 MB. `tracemalloc` measures allocations and can be
reset between runs. It traces numpy but not numba's internal allocations,
which suits what's being guarded here: the failure mode is a numpy-level
temporary the size of the data, not a kernel's own scratch.

**Throughput relative to scipy** — never absolute seconds. The same work is
timed through scipy in the same process and the *ratio* is recorded, which
cancels most of the difference between a fast laptop and a shared CI runner.
It's still the noisiest thing here, which is why the timing gates carry a
much looser margin (4×) than the memory ones (2×) and the layout ones (1.1×).

Every case runs in its own subprocess, because measurement state and JIT
warm-up would otherwise leak between them.

## Baselines

`baselines.json` holds a ceiling per gated metric. `--record` regenerates
them by multiplying a fresh measurement by that metric's margin. Only metrics
named in `margins` are gated; anything else a case returns is recorded for
context (e.g. `expanded_nnz_mb`, which says what a per-nonzero temporary
*would* have cost, so the gated number next to it can be read in proportion).

The checked-in ceilings were recorded on `main` before the v0.2 memory fixes
landed, so two of them are deliberately generous:

| metric | recorded on `main` |
|---|---|
| `minor_sum_peak_mb.peak_alloc_mb` | 66 MB, for a reduction on a 4M-nonzero array |
| `misaligned_matmul_peak_mb.peak_alloc_mb` | 204 MB, against a 16 MB array |

Both should drop by one to two orders of magnitude once the reduction and
misaligned-matmul fixes are in. **Re-record after those merge** — until then
these gate against getting worse, not against the current numbers being good.

## Adding a case

Write a function returning `{metric: value}` in `cases.py` and decorate it
with `@fast` (runs on every PR — keep it well under a minute) or `@slow`.
Add any new gated metric name to `margins` in `baselines.json`, then
`--record`.

## The larger suite

`--set slow` isn't wired into the PR gate: those cases build datasets an
order of magnitude bigger and take minutes, which is the wrong thing to put
in front of every push. They're meant for a scheduled workflow or a manual
run before a release. Wiring up that scheduled job is deliberately left as a
follow-up rather than guessed at here.
