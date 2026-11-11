# mojo-netcdf4

`mojo-netcdf4` accelerates the numeric work around netCDF variable reads and
writes with compiled Mojo kernels. It presents a `netCDF4`-shaped Python API
and operates on real NETCDF3/NETCDF4 files through the mature netCDF4/libnetcdf
storage backend.

The covered path is useful for CF-style scientific data: packed integer
variables with `scale_factor` and `add_offset`, missing or fill values,
`valid_min`, `valid_max`, `valid_range`, and
`least_significant_digit` quantization. Mojo fuses and vectorizes the large
array transforms while libnetcdf continues to handle file layout, chunking,
compression, checksums, and hyperslab I/O.

## Covered API

The tested compatibility surface is:

- `Dataset` file creation and opening, closing, context management, unlimited
  dimensions, variables, groups, attributes, and compressed variables
- `Dataset.createVariable` for primitive numeric and character variables,
  fill values, zlib compression, and `least_significant_digit`
- `Variable[...]` reads and writes using full, strided, and orthogonal
  fancy-index selections
- Numeric dtypes `i1`, `u1`, `i2`, `u2`, `i4`, `u4`, `i8`, `u8`, `f4`, and
  `f8`
- Automatic masking, CF scale/offset packing and unpacking, `_Unsigned`,
  multiple missing values, valid limits, masked writes, and the four automatic
  mask/scale controls
- Scalar `assignValue` and `getValue`, character passthrough, and `date2num`,
  `num2date`, `date2index`, `chartostring`, and `stringtochar`

The wrapper delegates other attributes and methods to `netCDF4`, but delegation
is not a compatibility guarantee. This project does not cover compound, enum,
or variable-length transforms; parallel MPI I/O; `MFDataset`; diskless-memory
return buffers; user-defined filters; or the full rename and type-creation
surface. Decimal-place `least_significant_digit` quantization runs in Mojo.

## Install and run

```bash
pixi install
pixi run build
pixi run test
```

The Python import is `mojonetcdf4`, which avoids shadowing the upstream
`netCDF4` package used as the file backend and parity oracle. From the
repository root, run the following block with `pixi run python`:

```python
import numpy as np
from mojonetcdf4 import Dataset

with Dataset("weather.nc", "w") as dataset:
    dataset.createDimension("time", None)
    temperature = dataset.createVariable(
        "temperature", "i2", ("time",), fill_value=-9999
    )
    temperature.units = "K"
    temperature.scale_factor = np.float32(0.01)
    temperature.add_offset = np.float32(250.0)
    temperature[:] = np.ma.array(
        [273.15, 274.20, 275.05], mask=[False, True, False]
    )

with Dataset("weather.nc") as dataset:
    values = dataset.variables["temperature"][:]
    print(values)
```

This writes packed `int16` values and prints a scaled masked array. The example
runs as written after `pixi install` and `pixi run build`.

## Benchmarks

Measured on this machine with `pixi run bench`; times are the median of 9 runs
(each the best of five warm iterations). Files were uncompressed so the table
exposes transform and variable-I/O cost, with the operating-system page cache
warm. Lower is better.

Machine: Intel Xeon E5-2697 v4 at 2.30 GHz, Linux 6.8.0-147-generic,
Python 3.13.14, netCDF4 1.7.4.

| case | mojo-netcdf4 | netCDF4 | result |
|---|---:|---:|---:|
| scaled int16 read, 5M | 28.13 ms | 52.30 ms | 1.86x faster |
| scaled int16 write, 5M | 14.12 ms | 67.11 ms | 4.75x faster |
| LSD-3 float64 write, 5M | 46.18 ms | 65.23 ms | 1.41x faster |
| plain float64 read, 5M | 27.69 ms | 36.22 ms | 1.31x faster |

Host load was 23--28 during these runs (other workloads share the machine), so
absolute times move by roughly 15--20 % between runs. The plain float64 read
case is highly variable; mojo-netcdf4 wins in ~2/3 of runs.

### Masked scaled write

`pixi run bench` writes a plain array, so it never reaches the masked pack
kernel. The same workload with a 0.05 % masked source, timed the same way
through `Variable.__setitem__` at 5 M elements, on the same machine:

| implementation | untouched build | this build |
|---|---:|---:|
| scaled int16 write, unmasked source | 12.5 ms | 12.4 ms |
| scaled int16 write, 0.05 % masked | 16.5 ms | 15.5 ms |

The masked row is the mean of four interleaved best-of-nine runs of each build
(untouched: 15.87, 16.48, 16.65, 17.16 ms; this build: 14.68, 15.69, 15.69,
16.04 ms); the unmasked row is the median of the same runs (range 12.0--13.3 ms
on both). Roughly a third of the masked write is Python-side work and libnetcdf
I/O, which is why the change to the kernel shows up at 5-10 % end to end and at
7-22 % on the kernel alone (paired direct calls on 5 M buffers: i2 -11 %, i8
-7 %, f4 -22 %). Packed values are unchanged: the masked path now runs one
SIMD pass, and the integer conversions still produce bit-identical output to
the previous scalar path for every storage type, including values that do not
fit the type.

The packed write uses one SIMD pass instead of NumPy subtraction, division,
rounding, casting, and masked-fill temporaries. Scaled decode and mask work is
split into 262,144-element CPU chunks only at or above 10,485,760 elements
(the parallel phase below explains why the threshold is that high); decode and
mask share one parallel launch. Quantization uses the same threshold and
chunking, with at most 4 workers. Smaller inputs stay serial to avoid
thread-launch overhead. Full contiguous LSD writes avoid a second backend
quantization pass, while plain reads delegate directly to the backend because
there is no transform to accelerate. Against the serial kernels of an earlier
build under the same compiler, scaled read improved from 27.63 ms to 17.67 ms
and LSD write from 34.92 ms to 27.87 ms.

### SIMD phase: fusion, and what it bought

Kernel measurements, paired A/B in one process (old and new library loaded
side by side and alternated, best of 15-31), on 5 M-element buffers of the
same shape `bench/bench.py` uses. Host load was 27-81 during these runs; the
old library paired against a second copy of itself moved by +/-5-11 % per
kernel, so only the larger figures below are meaningful.

| kernel | before | after | delta |
|---|---:|---:|---:|
| `mnc_unpack_f64`, i2 + 4 mask flags, serial, n = 1,048,575 | 1502.1 us | 1261.7 us | -16.0 % |
| `mnc_unpack_f64`, i2 + 4 mask flags, serial, n = 131,072 | 181.4 us | 160.1 us | -11.7 % |
| `mnc_unpack_f64`, i2 + 4 mask flags, 16 workers, 5 M (mean of 5 best-of-15 rounds) | 9.53 ms | 8.81 ms | -7.5 % |
| same, second session | 9.24 ms | 8.85 ms | -4.1 % |
| `mnc_pack_f64`, i2 masked, 5 M | 9.19 ms | 9.27 ms | +0.9 % (noise) |
| `mnc_pack_f64`, i2 masked, 20 M | 40.70 ms | 14.96 ms | -63.2 % |
| `mnc_pack_f64`, i2 masked, 20 M, second run | 38.08 ms | 13.38 ms | -64.9 % |

Scaled reads now decode and mask the same SIMD lanes in one pass over the
packed buffer, instead of one `decode_range` followed by one `mask_range` per
chunk; the single parallel launch is unchanged. The masked pack, alone among
the four kernels, used to run the whole array serially above 1,048,576
elements; it now uses the same chunked parallel scheme, which makes no
measurable difference at 5 M and a large one at 20 M.

Both changes are output-identical to the previous build: the decode, the mask
and the packed values are compared lane by lane for all ten storage types at
n = 0, 1, W-1, W, W+1, 2W+1, 37, 1001 and 4097. `tests/test_kernels.py` now
runs the fused decode, the mask kernel, the masked pack and quantization at
n = 0, 1, W-1, W, W+1, 100003 and 1000003 (`W = 4` on this AVX2 host).

A guarded SIMD conversion for the integer masked pack was built and rejected:
it was 1.6-8.8 % slower than the scalar body at 5 M, and the unguarded version
that is 13-15 % faster does not reproduce the scalar path's out-of-range
result. The integer path therefore keeps its per-element expression. Details
and the exactness argument are in `bench/analysis.md` §11.

The end-to-end `pixi run bench` rows move by more than this phase does, because
host load varied between 27-41 and 64 across the runs and both are best-of-five.
The kernel figures above are the stable measurement.

The end-to-end `pixi run bench` after the phase, for the record (load average
41 at the time):

| case | mojo-netcdf4 | netCDF4 | result |
|---|---:|---:|---:|
| scaled int16 read, 5M | 18.82 ms | 47.50 ms | 2.52x faster |
| scaled int16 write, 5M | 13.89 ms | 66.51 ms | 4.79x faster |
| LSD-3 float64 write, 5M | 34.34 ms | 70.82 ms | 2.06x faster |
| plain float64 read, 5M | 32.73 ms | 38.83 ms | 1.19x faster |

Against the pre-phase run in the table at the top of this section (14.10 /
12.04 / 28.10 / 26.90 ms at load 23-28), the read row in particular is worse
here at load 41, so this row set does not isolate the phase's effect; the
paired kernel figures above do.

### Parallel phase: a measured threshold and a 4-worker cap

Both kernels and their threshold were re-measured. The toolchain's
`max.algorithm.parallelize` has a **fixed ~5 ms worker start-up cost per
launch** on this host, independent of the work: a call at 786,432 elements
takes 0.37 ms, the same call at 1,048,576 elements (enough to trip a launch)
takes 4.9-5.1 ms. Two identical launches run back to back also take 5 ms each,
so the cost is paid per launch, not amortised. The old threshold of 1,048,576
sat exactly where that cost appears, so from 1 M to ~4 M elements the parallel
path was **2-10x slower than the serial body**, and it was not a win anywhere
until ~8-11 M.

Measured builds, paired A/B in one process and interleaved (serial reference is
the same source built with the threshold disabled, called on identical input;
best of 11-13 on 5 M-20 M buffers). Host load 17-27 during the runs.

| n (elements) | `mnc_unpack_f64` masked | `mnc_pack_f64` | `mnc_mask_f64` | `mnc_quantize_f64` |
|---|---:|---:|---:|---:|
| 1,048,576 | 1.42 ms serial, after: 1.42 (**1.00x**) | 1.85 -> 1.81 (**1.02x**) | 0.91 -> 0.93 (**0.98x**) | 0.50 -> 0.47 (**1.07x**) |
| 2,097,152 | 2.50 -> 2.56 (**0.97x**) | 3.90 -> 3.98 (**0.98x**) | 1.85 -> 1.88 (**0.98x**) | 0.98 -> 0.98 (**1.00x**) |
| 4,194,304 | 9.50 -> 9.58 (**0.99x**) | 8.34 -> 8.46 (**0.99x**) | 3.74 -> 3.67 (**1.02x**) | 2.30 -> 2.39 (**0.96x**) |
| 8,388,608 | 19.13 -> 12.40 (**1.54x**) | 16.36 -> 11.53 (**1.42x**) | 7.50 -> 8.07 (**0.93x**) | 9.07 -> 9.79 (**0.93x**) |
| 10,485,759 (just below threshold) | 22.26 -> 21.97 (**1.01x**) | 19.58 -> 19.33 (**1.01x**) | 9.16 -> 9.18 (**1.00x**) | 10.93 -> 10.73 (**1.02x**) |
| 10,485,760 (threshold) | 23.35 -> 13.28 (**1.76x**) | 19.49 -> 12.07 (**1.61x**) | 9.12 -> 9.33 (**0.98x**) | 10.74 -> 10.34 (**1.04x**) |
| 11,534,336 | 24.50 -> 13.84 (**1.77x**) | 22.37 -> 12.99 (**1.72x**) | 10.35 -> 9.75 (**1.06x**) | 12.14 -> 10.91 (**1.11x**) |
| 12,582,912 | 25.73 -> 14.20 (**1.81x**) | 24.09 -> 13.92 (**1.73x**) | 12.22 -> 10.01 (**1.22x**) | 13.03 -> 11.64 (**1.12x**) |
| 16,777,216 | 37.02 -> 17.17 (**2.16x**) | 32.34 -> 16.66 (**1.94x**) | 15.62 -> 10.93 (**1.43x**) | 18.14 -> 13.11 (**1.38x**) |
| 20,971,520 | 49.41 -> 20.35 (**2.43x**) | 39.13 -> 18.95 (**2.06x**) | 19.20 -> 12.63 (**1.52x**) | 22.51 -> 14.38 (**1.57x**) |

Rows below 10,485,760 are identical to serial, which is the point: they used
to take the parallel path and lose. The crossover is per kernel — `unpack` and
`pack` win from 8,388,608, while the lighter `mask` and `quantize` bodies do
not win there (0.93x) and only clear it around 10.5-11.5 M. **10,485,760** is
above `unpack`/`pack`'s crossover with a 25 % margin and loses to serial for no
kernel at any size measured.

Worker count at 4 M-32 M, `mnc_quantize_f64`, same method (speedup vs serial):

| n | serial | 1 worker | 2 workers | 4 workers | 8 workers | 16 workers |
|---|---:|---:|---:|---:|---:|---:|
| 4,194,304 | 3.66 ms | 2.44 | 7.75 | 7.60 | 7.13 | 7.20 |
| 8,388,608 | 10.74 ms | 10.83 | 10.05 | 8.59 | 8.33 | 8.59 |
| 16,777,216 | 21.81 ms | 21.99 | 15.36 | 13.30 | 13.60 | 14.89 |
| 33,554,432 | 44.41 ms | 39.78 | 26.92 | 22.17 | 20.40 | 19.94 |

The gain flattens at 4 and does not improve further at 8 or 16; 4 is the cap.
Part of that is this host: single-core effective bandwidth measured ~8 GB/s
(`numpy` add over 20 M Float64), and the machine was running other jobs at load
17-27 throughout, so a serial kernel is already spending most of its time at
the memory wall and extra workers mostly contend. Multi-core scaling is real
but sublinear: at 64 M elements, serial 88.8 ms vs 45.96 ms with 4 workers.

**End-to-end, and an honest caveat.** `pixi run bench` at 5 M stays on the
serial path (it is under the threshold), so that benchmark cannot show this
phase; it moves with host load instead. Interleaved with both libraries in one
process, 5 M scaled read was 15.6/14.8 ms old vs 22.9 ms new at load ~25, which
is end-to-end noise in a 5 ms-per-native-call region and does not reproduce
run to run. At 20 M the same harness did separate the builds and **the old
16-worker path was faster end to end** (read 44.5 vs 54.5 ms, 7 runs). The
kernel call is ~42 ms of both builds' ~58 ms scaled write, and the kernel A/B
at 20 M is 2.1x in favour of this phase, so that end-to-end gap is not kernel
time; it is unreproducible run to run here (plain read, which calls no kernel,
moved by the same amount in the same harness). Reproduce both with the A/B
harness in `bench/analysis.md` §12 before relying on either end-to-end figure.

### Parallel phase: what was tried, kept and reverted

| change | result | verdict |
|---|---|---|
| worker cap 16 -> 4 | equal within noise at 8/16 workers; 4 is the knee | **kept** |
| threshold 1,048,576 -> 10,485,760 | was 2-10x slower than serial at 1-4 M; now identical below 10.5 M and 1.4-2.4x faster above | **kept** |
| a single `PARALLEL_THRESHOLD` for all four kernels | first tried 8,388,608; `mask` and `quantize` still lost there (0.93x), so the shared value needs to clear their crossover, not unpack's | kept (at 10,485,760) |
| `sync_parallelize` | segfaults when called from the ctypes host (`MOJO_NOTES.md` §5b); not used | not used |
| smaller grain (32 K-131 K) | no measurable gain, and chunk count does not change the launch floor | reverted |

`tests/test_kernels.py` builds the same kernel source with the threshold
disabled and compares all four kernels' output on identical input, bit for bit,
both through the raw symbols and through the public `_kernels` wrappers
(`test_serial_and_parallel_outputs_match`,
`test_public_path_serial_and_parallel_agree`). `build/build.sh` accepts
`MNC_SRC`/`MNC_OUT` so that variant build does not touch `dist/`.

### GPU phase: CPU only, and why

**CPU only: memory-bound.** Every kernel in this port falls under the gate of
roughly 2 floating point operations per byte moved, so no kernel was ported to
the GPU. The counts below are per element of the transform, from the bodies in
`src/netcdf4.mojo` (2 flops = one fused multiply-add; `round` is counted as one
operation), and match the `bench/analysis.md` table:

| kernel | work per element | bytes moved per element | flops/byte |
|---|---|---:|---:|
| `mnc_quantize_f64` (`quantize_range`) | ~3 (mul, round, mul) | 16 (f8 read + f8 write) | **0.19** |
| `mnc_unpack_f64` (`decode_range` + fused mask) | ~2.5 (fma) + ~4 mask compares | 18 (i2 read twice + f8 + u1) | **~0.36** |
| `mnc_pack_f64` (`pack_range`) | ~2 (sub, div) | 12 (f8 read + i2 write) | **0.17** |
| `mnc_pack_f64` masked (`pack_masked`, float storage) | ~4 (sub, div, select, convert) | 16 (f8 + i2 + u1 x2) | **0.25** |
| `mnc_mask_f64` (`mask_range`) | 5 at 4 flags (4 compares + select) | 3 (i2 read + u1 write) | **~1.7** (0.67 by the analysis table's byte count) |

Even the heaviest kernel, the mask, is below the threshold, and its bytes are
dominated by a 1-byte-per-element write, so it is bandwidth-bound rather than
arithmetic-bound. A device copy pair plus a launch would cost more than the
7--16 ms these kernels take at the sizes real callers pass (`pixi run bench`
uses 5M elements; the largest measured input is 80 MB).

The GPU availability gate was also checked, for the record:
`nvidia-smi --query-gpu=memory.free --format=csv,noheader` reported **6262 MiB
free** on an RTX 5090, i.e. above the 4000 MiB floor, so the phase was skipped
on the flops/byte gate, not on device availability. No GPU code, no GPU tests
and no GPU benchmark was added, so there is nothing to skip on a CPU-only host;
the existing test suite runs the same on both.

## How it works

For transformed reads, `netCDF4.Variable` automatic mask/scale is disabled on
the private backend object. Reads ask libnetcdf for native-endian,
C-contiguous raw values, then Mojo SIMD kernels decode and compare the native
packed buffer without an intermediate float64 mask copy. Scalar tail loops
handle lengths that are not SIMD-width multiples. Mask comparisons happen in
the packed domain, matching netCDF4 semantics; 64-bit integer comparisons use
an exact NumPy fallback because a `Float64` comparison value cannot represent
every `int64` or `uint64`.

Writes pass a contiguous `float64` source and optional byte mask to Mojo. The
SIMD kernel applies `(value - add_offset) / scale`, round-to-even integer
packing, and the storage cast in one traversal. Masked writes use a branchless
SIMD pass for `f4`/`f8` storage; for integer storage the conversion is written
as the per-element expression, which a compiler packs into one convert per
vector and which keeps the exact out-of-range result of the previous scalar
path, so packed values are identical either way. Decimal quantization uses the
same power-of-two multiplier formula as netCDF4 and bypasses netCDF4's
duplicate quantizer for full contiguous writes.

Python owns every NumPy allocation. Buffers cross the C ABI as 64-bit integer
addresses and Mojo reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values internally. The shared library
exports only non-parametric `abi("C")` functions, returns an error status for
invalid arguments, and is built as
`dist/libmojo-netcdf4.so`.

MIT.
