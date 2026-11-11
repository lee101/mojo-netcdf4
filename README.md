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

Measured on this machine with `pixi run bench`; times are the best of five warm
iterations.
Files were uncompressed so the table exposes transform and variable-I/O cost,
with the operating-system page cache warm. Lower is better.

Machine: Intel Xeon E5-2697 v4 at 2.30 GHz, Linux 6.8.0-147-generic,
Python 3.13.14, netCDF4 1.7.4.

| case | mojo-netcdf4 | netCDF4 | result |
|---|---:|---:|---:|
| scaled int16 read, 5M | 14.10 ms | 45.92 ms | 3.26x faster |
| scaled int16 write, 5M | 12.04 ms | 69.82 ms | 5.80x faster |
| LSD-3 float64 write, 5M | 28.10 ms | 84.15 ms | 2.99x faster |
| plain float64 read, 5M | 26.90 ms | 25.39 ms | 1.06x slower |

Host load was 23--28 during these runs (other workloads share the machine), so
absolute times move by roughly 15--20 % between runs and the ratios move with
them (the scaled-read ratio measured between 2.0x and 3.3x across this
session's runs of the same builds). Times are best-of-five, as reported by
`bench/bench.py`.

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
split into 262,144-element CPU chunks only at or above 1,048,576 elements;
decode and mask share one parallel launch. Quantization uses the same threshold
and chunking. Smaller inputs stay serial to avoid thread-launch overhead. Full
contiguous LSD writes avoid a second backend quantization pass, while plain
reads delegate directly to the backend because there is no transform to
accelerate. Against the serial kernels of an earlier build under the same
compiler, scaled read improved from 27.63 ms to 17.67 ms and LSD write from
34.92 ms to 27.87 ms.

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

There is no GPU path. These transforms perform only about 0.15--0.3 floating
point operations per byte moved, well below the roughly 2 FLOP/byte threshold
where device transfer and launch costs can be justified.

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
