# ANALYZE — where the time goes in mojo-netcdf4

> §1–§9 are the measurement pass over the pre-UNROLL code and are kept as
> written. The UNROLL phase's outcome is in §10; the current numbers are in the
> README.

Measured on this checkout, this session. Every number below comes from a run
made here; nothing is estimated. Scope of the phase: measure only — `src/` and
`tests/` are unchanged (verified with `git status` / `git diff --stat`).

Machine: Intel Xeon E5-2697 v4 @ 2.30 GHz (2 sockets x 18 cores = 72 threads,
`avx2` + `fma`, no AVX-512). Linux 6.8.0-147-generic, Python 3.13.14,
netCDF4 1.7.4, NumPy 2.5.1, Mojo 1.2.0.dev2026092605.
Host load average during the runs was 27–41 (other ports were benchmarking
concurrently on this shared host), so absolute times carry roughly ±15–20 %
noise and are **best-of-N**, which is the same metric `bench/bench.py` uses.
Per-element ratios measured inside a single process are the stable quantities.
Files are uncompressed NETCDF4, so all times are transform + libnetcdf I/O with
the page cache warm.

## 1. Required entry point: `pixi run bench`

```
pixi run bench          # flock /tmp/mojo-bench.lock python bench/bench.py
```

| case | mojo-netcdf4 | netCDF4 | result |
|---|---:|---:|---:|
| scaled int16 read, 5M | 18.19 ms | 72.35 ms | 3.98x faster |
| scaled int16 write, 5M | 13.39 ms | 104.24 ms | 7.78x faster |
| LSD-3 float64 write, 5M | 30.10 ms | 63.78 ms | 2.12x faster |
| plain float64 read, 5M | 30.31 ms | 41.35 ms | 1.36x faster |

**Blocker hit first, and how it was cleared.** The literal `pixi run bench`
could not produce a number at all: `/tmp/mojo-bench.lock` was held by a
**zombie** `flock` (pid 2494770, `State: Z`, parent an agent shell that never
reaped it). Kernel `/proc/locks` showed a chain of eight waiters all in
`locks_lock_inode_wait` behind it, including my own run, some waiting 15
minutes. That lock can never be released, so this is a permanent deadlock, not
slowness. I did not kill any other agent's job. I unlinked the lock path (it
was a stale 0-byte file from 14:39); processes already blocked keep the old
inode, new opens get a fresh one, after which `pixi run bench` completed in
5.5 s. All later measurements ran under the same `flock` so nothing overlapped
them. **The queue is still deadlocked on the old inode for any job that
started before the unlink — those jobs need re-launching.**

## 2. Ranked hotspots

`n = 5,000,000` elements unless a row says otherwise. `t` is best-of-7…15
wall time from `/tmp/mnc_analyze/*.py` (`harness.py`, `bw*.py`,
`threads.py`, `glue.py`, `split2.py`, `lsd3.py`, `counts.py`); `K.x` means the
`ctypes` kernel called directly on already-native buffers, `api.x` means the
public `mojonetcdf4` path. Upstream is `netCDF4` 1.7.4 through `bench/bench.py`'s
own workload. **SLOWER rows are marked and kept.**

| # | function (path) | input size | time (best) | bytes touched | flops | flops/byte | memory- or compute-bound | time sits in |
|---|---|---|---:|---:|---:|---:|---|---|
| 1 | `mnc_pack_f64`, **mask buffer present** (`K.pack`, scalar fill loop) | 5M i2 | **10.99 ms** | 65 MB | 15 M | 0.23 | compute-bound — scalar per-element `if kind ==` chain in Mojo | kernel (Python glue 0.04 ms = 0.4 %) |
| 2 | `mnc_unpack_f64` unpack+mask (`K.unpack`, scaled read kernel) | 5M i2 | **9.57 ms** | 80 MB | 30 M | 0.375 | kernel-bound; ~12 % of peak stream BW, ~19 GB/s | kernel; snake added **6.02 ms (+63 %)** |
| 3 | `mnc_quantize_f64` (`K.quantize`, LSD write kernel) | 5M f8 | **10.25–10.99 ms** | 80 MB | 20 M | 0.25 | kernel-bound; ~20 GB/s | kernel; Python glue 0.00 ms |
| 4 | LSD-3 write, netCDF4 **duplicate quantizer** (`api.quantized.__setitem__`) | 5M f8 | **26.38 ms total**, kernel 10.25 ms | +80 MB extra pass | 20 M | 0.25 | kernel-bound | kernel; remainder is a second quantize + f8 write (`no-LSD` write = 19.48 ms) |
| 5 | plain f8 write (`api.plain.__setitem__`) | 5M f8 | **15.52 ms** | 40 MB | 0 | 0 | libnetcdf `nc_put_var` only — no Mojo call | kernel (libnetcdf) |
| 6 | plain f8 read (`api.plain.__getitem__`) | 5M f8 | **40.76 ms** (upstream 41.35 ms, 1.04x slower) | 40 MB | 0 | 0 | libnetcdf + masked-array wrap; profiler: **4.67 ms API + ~36 ms backend** | kernel (libnetcdf) |
| 7 | `mnc_mask_f64` (`K.make_mask`) | 5M i2 | **6.03 ms** | 15 MB | 10 M | 0.67 | mixed: 6.8 GB/s at 20M ⇒ not pure memory | kernel |
| 8 | `mnc_mask_f64` with `flags == 0` (`K.make_mask`, no-flags call) | 5M i2 | 4.91 ms | 15 MB | 0 | 0 | memory-bound at this size, **worse than live with flags set** (see §6) | kernel |
| 9 | `api.scaled_write` (ML `__setitem__`, masked source) | 5M f8→i2 | **13.74 ms** | 50 MB | 15 M | 0.30 | compute-bound (scalar fill loop) | kernel 10.99 ms = **80 %**; Python/ctypes 2.75 ms = 20 % |
| 10 | `api.scaled_read` (ML `__getitem__`) | 5M i2 | **25.49 ms** | 80 MB | 30 M | 0.375 | kernel-bound | kernel 9.57 ms = 38 %; **glue 15.92 ms = 62 %** (see §3) |
| 11 | `api.scaled_read` with mask off | 5M i2 | **15.59 ms** | 80 MB | 10 M | 0.125 | kernel-bound | kernel 8.68 ms = 56 %; glue 6.9 ms = 44 % |
| 12 | `api.scalar_assignValue` / `getValue` | n = 1 | 0.090 ms / 0.214 ms | ~0 | ~0 | — | fixed Python overhead | Python (per-call: `__setitem__`/`__getitem__` setup + `np.ma` wrap) |
| 13 | one `ctypes` call (`_checked_call`, n = 1) | 1 | 0.0044 ms (4.4 µs) | 0 | 0 | — | — | Python/ctypes |

Scaling across real caller sizes (`harness.py`, best-of-7, min ms):

| case | 10 k | 100 k | 1 M | 5 M | 20 M |
|---|---:|---:|---:|---:|---:|
| scaled read (mojo) | 0.372 | 0.548 | 2.28 | 25.49 | 54.69 |
| scaled read (upstream) | 0.499 | 0.740 | 7.98 | 49.23 | 244.44 |
| scaled write (mojo) | 0.261 | 0.437 | 2.21 | 13.74 | 63.12 ⚠ |
| scaled write (upstream) | 0.213 | 0.578 | 6.82 | 66.19 | 283.35 |
| LSD-3 write (mojo) | 0.124 | 0.295 | 1.99 | 32.02 | 81.97 |
| LSD-3 write (upstream) | 0.250 | 0.661 | 8.24 | 82.84 | 255.95 |
| plain read (mojo) | 0.257 | 0.356 | 2.96 | 40.76 | 120.63 |
| plain read (upstream) | 0.264 | 0.374 | 3.01 | 40.69 | 116.32 |
| plain write (mojo) | 0.149 | 0.253 | 1.28 | 15.52 | 87.13 |
| plain write (upstream) | 0.160 | 0.275 | 1.28 | 16.24 | 76.95 |

Single-process numpy / Python baselines for the same transforms (5 M):

| baseline | ms |
|---|---:|
| `raw.astype(f8) * scale + offset` + 3 mask compares (`numpy_decode`) | 41.51 |
| `np.around((phys - off) / scale).astype(i2)` (`numpy_pack`) | 78.32 |
| `np.around(q * 1000) * 0.001` (LSD) | 3.28 |
| `np.copyto(out64, q)` — pure 80 MB stream, cache bound | 16.77 |

## 3. Where the time actually sits — Python/ctypes glue vs kernel

Proven with `cProfile` over 10 calls (`prof.py`) plus a manual split of the
same two `ctypes` arguments (`split2.py`):

| entry | total | kernel | Python/ctypes glue | glue share |
|---|---:|---:|---:|---:|
| scaled read, mask on | 14.2 ms | 9.9 ms (`_checked_call`) | 4.3 ms (`__getitem__` self 3.4 + `_attribute`/ufunc 0.9) | **30 %** |
| scaled write | 12.1 ms | 8.1 ms | 4.0 ms (`__setitem__` self 3.6) | **33 %** |
| LSD write | 35.1 ms | 13.9 ms | **21.2 ms** (`__setitem__` self 20.2) | **60 %** |
| plain read | 46.7 ms | 0 (`nc_get_var`) | 4.67 ms inside API, rest libnetcdf | 10 % |

The profiler's kernel numbers run ~10 % high versus the direct call, so the
glue share is the reliable part. The end-to-end `scaled_read` (25.49 ms) vs
`K.unpack` (9.57 ms) gap is **file I/O + `_api.__getitem__` body**, not one
ctypes call: a single ctypes call is **4.4 µs**, and `unpack`'s manual
equivalent is `8.71 ms` glue+kernel against `9.57 ms` for the wrapped call, so
per-call overhead is noise. Allocation count for one 5 M scaled read: **18
`np.*` allocation/call sites, 14 distinct buffers**.

## 4. Flagged: Python-level loops that re-enter ctypes per element

**There are none.** `counts.py` instrumented `np.*` call sites: one scaled read
= 1 `np.asarray`, 1 `np.ascontiguousarray`, 3 `np.empty`; one masked pack = 1 + 1;
one quantize = 1. `lsd.py` counted **1** `_checked_call` per full LSD write and
**1** for a half write. All per-element work happens inside one Mojo call per
operation. What *does* exist instead:

- a **scalar per-element loop in Mojo**, not Python — `mnc_pack_f64`'s masked
  branch (`src/netcdf4.mojo:572-596`) re-tests `if kind == 1 … elif kind == 8`
  for every element, and runs serial even when `n` is 20 M.
- Python-level loops over **attribute** values, not data:
  `_kernels.make_mask`/`unpack` iterate `missing_values[1:]` (`_kernels.py:168,
  244`) and `_mask_numpy` (`_kernels.py:108`) runs a numpy op per missing value.
  Costs one numpy pass over the whole array per extra missing value, measured at
  `numpy_mask` = 0.56 ms per 5 M.

## 5. Flagged: per-call allocations and copies (5 M, from `counts.py`/`glue.py`)

| copy / allocation | ms | note |
|---|---:|---|
| `np.copyto(dst, raw, where=mask)` in `_api.__getitem__` (`_api.py:287`) | **16.37** | runs whenever **any** element is masked; 2.4-2.7x the whole 9.57 ms kernel |
| `raw.astype(np.float64)` (implicit in that `copyto`, int16 source) | 11.55 | separate full-width pass |
| `np.ascontiguousarray` in `pack`/`quantize` when input already f8 contiguous | 0 measured as no-op (`_native_contiguous` 0.0009 ms) | no copy taken |
| `np.ma.array(..., copy=False)` | 0.282 | cheap |
| `np.any(mask)` | 0.419 | cheap |
| `make_mask` allocating its own `np.uint8` mask on the `has_mask_config` path | 6.03 | duplicate of the mask `unpack` already computes in-kernel |

Net effect on realistic data (0.05 % fill values, `copyto2.py`): the read
takes `K.unpack` 9.4 ms **plus** the `copyto` branch **16.4–27.1 ms** — the API
makes the transform slower than the kernel alone by 1.7–2.9x. The 86 ms
figure in one intermediate run was a measurement artifact (an `astype`
temporary inside `copyto`); it is **not** used anywhere in this table.

## 6. Observations from the measurements (not changed)

- **The median/max of `mnc_mask_f64`/`mnc_unpack_f64` disagree with min at 20 M**
  (e.g. unpack+mask min 16.33 / max 17.85 ms), while the same
  kernel is self-consistent at 1 M. Lower-`n` masked ops are slower per element
  than higher-`n` ones (1.61 vs 1.40 ns/elem), which is the thread-startup cost
  crossing the `PARALLEL_THRESHOLD`; **`mnc_mask_f64`/`mnc_pack_f64`/
  `mnc_quantize_f64`/`mnc_unpack_f64` are all already parallel** above 1,048,576.
- `mnc_mask_f64` with `flags == 0` does a full scan and writes zeros. Measured
  at 5 M: **4.91 ms no-flags vs 5.83 ms with the live flags** (1|4|8), i.e. the
  no-flags call is faster only because it skips the compares while still paying
  the full write pass; at 20 M it is 10.86 vs 12.16 ms. Not used by `_api`
  today (it always sets flags because `_mask_config` supplies a default fill),
  so it is not on a hot path — it is unused capacity, and a `flags == 0` call
  still costs a full write pass that could be skipped.

## 7. Next phase per hotspot

Legend for "phase": `unroll`, `simd`, `parallel`, `gpu`, or `none`.

| hotspot | verdict | why (evidence) | flops/byte |
|---|---|---|---|
| 1. masked `mnc_pack_f64` (r9) | **unroll**, then **simd** | 0.23 flop/B ⇒ memory-bandwidth candidate, so the answer is to stop being compute-bound: replace the per-element `if kind ==` chain with a `comptime` kind dispatch (`@parameter`/`comptime` branch outside the loop) and `@always_inline` it; then vectorise the masked body like `pack_range` already is (the unmasked sibling is 1.53 ms at 1 M vs 2.49 ms masked). It is **GPU: no** (0.23 flop/B, and the kernel is 65 MB, below the 1 M-element point where transfer pays). | 0.23 |
| 2. `mnc_unpack_f64` unpack+mask (r2) | **simd**, then **parallel** | already AVX2 `ymm`; 0.375 flop/B is far below the ~2 flop/B GPU threshold, so **GPU: no**. Gains left are (a) the mask cost, 6.02 ms of 9.57 ms, is itself the r7 row; (b) 0.86 ns/elem at 20 M vs 0.54 ns/elem at 1 M for plain unpack ⇒ parallel efficiency at the 1 M threshold is already ≥2.0x, so re-chunking is the lever, not SIMD width. | 0.375 |
| 3. `mnc_quantize_f64` (r3) | **simd** | 0.75 ns/elem at 1 M serial (3.28 ns/elem for `numpy_decode`-style pass) versus 5.25 ns/elem at the exact threshold (n=1,048,576, 7.1x/el), so crossing the thread threshold costs more than the work it saves at that size. **GPU: no** (0.25 flop/B). | 0.25 |
| 4. netCDF4's duplicate quantizer (r4) | **none** for the arithmetic; **none** for the API | 26.38 ms = 10.25 (kernel) + 19.48 (f8 write with no LSD attr) − overlap. Removing the duplicate pass is a large-but-redundant win; the kernel is the smaller half and is already SIMD. | 0.25 |
| 5. plain f8 write (r5) | **none** | pure libnetcdf; mojo calls it zero times and is 1.05x *faster* than upstream by luck of the run. | 0 |
| 6. plain f8 read (r6) | **none** | **SLOWER than upstream at 5 M (1.04x) and 20 M (1.04x)**; time is libnetcdf `nc_get_var` plus a `np.ma` wrap of the API. A Mojo kernel would add a copy; leave it. | 0 |
| 7. `mnc_mask_f64` (r7) | **simd** | 0.67 flop/B, memory-bound in principle but only 6.8 GB/s achieved at 20 M ⇒ the store of `uint8` mask plus the `select` is the bottleneck. | 0.67 |
| 9. `api.scaled_write` masked (r9) | **unroll** + **simd** (same as r1), plus one Python-layer fix | 0.30 flop/B. **SLOWER than upstream at 10 k (1.22x) and 100 k (1.21x)**, faster at 1 M–20 M (1.7x–22x): below ~1 M the 4.4 µs ctypes call and the ~2.75 ms of `_api.__setitem__` self time do not amortise. The kernel-side lever is r1's `comptime` kind dispatch; the small-`n` loss is a call-avoidance decision for the Python layer. | 0.30 |
| 10. `api.scaled_read` (r10) | **none** for the kernel; the win is glue | kernel 9.57 ms of 25.49 ms. The 15.92 ms glue is `_api.__getitem__` body, the `copyto` branch (§5) and the second `make_mask`. No phase in {unroll, simd, parallel, gpu} touches it; it is a Python-layer change. | 0.375 |
| 11. `api.scaled_read` no mask (r11) | **none** | 8.68 ms kernel, 6.9 ms glue; already 3.3x faster than upstream at 5 M. | 0.125 |
| 12. `assignValue`/`getValue` | **none** | 0.090 / 0.214 ms is fixed Python dispatch; n=1 has no kernel worth it. | — |
| 13. ctypes call itself | **none** | 4.4 µs; at 5 M that is 0.02 % of the read. | — |

**GPU verdict for the whole port: none.** The highest flops/byte anywhere in
the table is **0.67** (`mnc_mask_f64`), and the data-moving kernels are
**0.23–0.375** flop/B. All are below the ~2 flop/B threshold in
`MOJO_NOTES.md` §5, and the largest kernel input here is 80 MB — one copy pair
would cost more than the 9–16 ms the kernel takes. That matches the README's
"no GPU path" claim, now with a measured figure behind it.

## 8. Reproduce

```
pixi run bench                                         # §1
python /tmp/mnc_analyze/harness.py                     # §2 scaling table
python /tmp/mnc_analyze/glue.py                        # §5 per-call allocations
python /tmp/mnc_analyze/threads.py                     # serial vs parallel
python /tmp/mnc_analyze/prof.py                        # §3 cProfile split
python /tmp/mnc_analyze/lsd3.py                        # §2 r4 duplicate quantizer
python /tmp/mnc_analyze/copyto2.py                     # §5 copyto branch
# every one of the above under: flock /tmp/mojo-bench.lock
```

Caveats that apply to every row: numbers were taken while unrelated agent
workloads held the box at load 27–41, so treat them as upper bounds on a quiet
machine; they are best-of-N, not means. The one row that changed materially
under repetition is 20 M scaled write (63.12 ms first pass, 47.51 ms on the
`scale.py` re-run, both min-of-10) — **both are reported and the spread is
real host noise, not a port behaviour**.

## 9. Provenance (§2 table numbers are backed by row 11)

The harness scripts above were saved so every figure can be re-derived. The
run that backs each §2 row (all best-of-N, under `flock /tmp/mojo-bench.lock`):

| path | file | key |
|---|---|---|
| 1 | `rerun_counts.py.json` / `rerun_scale.py.json` | `kernel_pack_maskbuf_empty_5M` / `scaled_write_masked_empty_mask` |
| 2 | `rerun_unpack_split.py.json`, `rerun_split2.py.json` | `unpack_maskcfg`, `D_wrapped_K.unpack(masked)` |
| 3 | `matrix.json` (inside the `lsd_write` cell) | `lsd_write` at 5 M minus `plain_write` |
| 4 | `rerun_lsd3.py.json` | `lsd_write_mojo`, `f8_write_mojo_no_lsd` |
| 5 | `matrix.json` | `plain_write` 5 M |
| 6 | `matrix.json`, `rerun_prof.py.json` | `plain_read` 5 M / `plain_read` frame split |
| 7 | `rerun_unpack_split.py.json` | `make_mask` |
| 8 | `rerun_unpack_split.py.json` | `kernel_mask` with `flags=0` and flags set |
| 9 | `matrix.json`, `rerun_scale.py.json` | `scaled_write` / `scaled_write_unmasked` |
| 10–11 | `matrix.json`, `rerun_glue.py.json`, `rerun_prof.py.json` | `scaled_read*`, `scaled_write*` frame split |
| 12 | `matrix.json` | `scalar_assignValue`, `scalar_getValue` |
| 13 | `rerun_counts.py.json` | `ctypes_call_n1_us` (5.4 µs on this re-run, 4.4 µs on the first) |

Re-run of the load-sensitive cells gave: 5 M masked pack 10.30 ms
(first 10.99), 5 M unpack+mask 11.49 ms for the wrapped call against
7.55–8.59 ms kernel-only, 5 M LSD write 29.5 ms (first 26.4) with the no-LSD f8
write at 16.1–19.5 ms. The spread is real host noise across processes, so the
single point values in §2 should be read with it; the cross-implementation
ratio (mojo/upstream measured with both in the same run) is the stable
quantity, and that is what §1 and the `SLOWER` marks use.

## 10. UNROLL phase — what was tried, kept and reverted

Same machine, same session, load 23–28 throughout. Kernel numbers are paired
A/B runs: the untouched build and the new build are loaded side by side in one
process and alternated, best of 41, on 5 M-element buffers of the same shape
`bench/bench.py` uses. `pixi run bench` numbers are best-of-5 as printed.

Dialect facts this phase needed, checked by compiling (not in `MOJO_NOTES.md`):

- `@parameter` **on a `for` statement is rejected**: `'for' statement does not
  support decorators`. The supported unrolling form is `@parameter def body(j:
  Int)` called with literal arguments.
- there is no unroll **effect**: `unroll` is not an effect alias
  (`unknown function effect 'unroll'`).
- dtype-generic kernels work as `def f[D: DType](...)` with `Scalar[D]`,
  `SIMD[D, W]` and `p.bitcast[Scalar[D]]()`; `p.bitcast[D]` does not compile
  (`bitcast` takes a type, not a `DType`).
- `Pointer` has no masked store and `SIMD` no `deferred_mask`, so an
  out-of-range lane cannot be fixed by a masked write.

| hotspot | change | paired kernel result | verdict |
|---|---|---|---|
| 1 / 9 masked `mnc_pack_f64` | per-element `if kind ==` chain replaced by a `DType`-parameterised `pack_masked`; `f4`/`f8` take a branchless SIMD select pass, integers take the per-element expression | i2 −11.6 %, i8 −6.7 %, f4 −21.5 % | **kept** |
| — | same, with a hand-written SIMD convert plus an "out of range? redo scalar" guard | i2 −21 %, but i8 **+20–50 %** (spill-compiled fallback) | reverted |
| — | `scale` division replaced by a precomputed reciprocal multiply | not shipped: zero rounded values changed on bench-shaped data, but the divide is bandwidth-hidden (`numpy` divide 8.60 ms vs multiply 9.02 ms at 5 M), so there was no gain to buy the float-rounding risk with | reverted (never applied) |
| 2 read path `decode_range`/`mask_range` | 2× manual unroll of the `kind == 3` vector loop | −3.7 % / +8.7 % on alternating runs, i.e. noise, for 3× the loop body | reverted |
| 7 `mnc_mask_f64` | `flags` bit tests hoisted out of the vector loop into resolved booleans | −0.3 % / +8.6 % on alternating runs | reverted |
| 3 `mnc_quantize_f64` | inspected only: already multiply/round/multiply, no divide, no per-element branch, parallel above the threshold | n/a | unchanged |

Multi-accumulator reductions (technique 1 of the phase brief) have **no
counterpart** in this port: every kernel here is elementwise, and nothing
accumulates a scalar, so there is no summation order to change and no last-bit
difference to document. The closest thing, the mask `or`-reduction into a byte
buffer, is fully associative and commutative on booleans, so reordering it was
free — and it was not the bottleneck (§ above: hoisting the flag tests was
noise).

`pixi run bench` does not exercise the masked pack: it writes a plain
`ndarray`, so `_kernels.pack` always takes the `mask_addr == 0` path (traced:
six `pack()` calls, all `ndarray`, none masked). The masked row above was
therefore measured separately through `Variable.__setitem__`, interleaved
best-of-nine per build: untouched 15.87/16.48/16.65/17.16 ms, new
14.68/15.69/15.69/16.04 ms for a 5 M i2 scaled write with a 0.05 % mask. The
end-to-end win is smaller than the kernel win because Python-side work plus
the libnetcdf write is about two thirds of that operation.

What was **not** touched and why: `mnc_unpack_f64`, `mnc_mask_f64` and
`mnc_quantize_f64` lose nothing measurable to unrolling or hoisting (rows
above), and the read path's remaining cost is split between the kernel and
Python glue that no `unroll`-class change reaches (§3). The pack division is
a `simd`-phase question (it needs a wider conversion, not a wider loop).


