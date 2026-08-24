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

Machine: Intel Xeon E5-2697 v4 at 2.30 GHz, Linux 6.8.0-136-generic,
Python 3.13.14, netCDF4 1.7.4.

| case | mojo-netcdf4 | netCDF4 | result |
|---|---:|---:|---:|
| scaled int16 read, 5M | 17.67 ms | 48.85 ms | 2.76x faster |
| scaled int16 write, 5M | 12.27 ms | 66.93 ms | 5.46x faster |
| LSD-3 float64 write, 5M | 27.87 ms | 60.76 ms | 2.18x faster |
| plain float64 read, 5M | 28.16 ms | 27.76 ms | 1.01x slower |

The packed write uses one SIMD pass instead of NumPy subtraction, division,
rounding, casting, and masked-fill temporaries. Scaled decode and mask work is
split into 262,144-element CPU chunks only at or above 1,048,576 elements;
decode and mask share one parallel launch. Quantization uses the same threshold
and chunking. Smaller inputs stay serial to avoid thread-launch overhead. Full
contiguous LSD writes avoid a second backend quantization pass, while plain
reads delegate directly to the backend because there is no transform to
accelerate. Against the serial kernels under the same compiler, scaled read
improved from 27.63 ms to 17.67 ms and LSD write from 34.92 ms to 27.87 ms.

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
packing, and the storage cast in one traversal; masked writes retain a fused
scalar path. Decimal quantization uses the same power-of-two multiplier
formula as netCDF4 and bypasses netCDF4's duplicate quantizer for full
contiguous writes.

Python owns every NumPy allocation. Buffers cross the C ABI as 64-bit integer
addresses and Mojo reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values internally. The shared library
exports only non-parametric `abi("C")` functions, returns an error status for
invalid arguments, and is built as
`dist/libmojo-netcdf4.so`.

MIT.
