"""End-to-end variable I/O benchmarks against netCDF4."""

from __future__ import annotations

import math
import os
import platform
import sys
import tempfile
import time

import netCDF4 as nc
import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojonetcdf4 as mnc  # noqa: E402


def timeit(function, repeat=5):
    function()
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def result(reference, mojo):
    ratio = reference / mojo
    return f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"


def main():
    rng = np.random.default_rng(2026)
    n = 5_000_000
    physical = np.ascontiguousarray(
        rng.normal(280.0, 12.0, n), dtype=np.float64
    )
    raw = np.rint((physical - 250.0) / 0.01).astype(np.int16)
    quantized = np.ascontiguousarray(rng.normal(size=n), dtype=np.float64)
    rows = []

    with tempfile.TemporaryDirectory() as directory:
        paths = {
            "mojo": os.path.join(directory, "mojo.nc"),
            "reference": os.path.join(directory, "reference.nc"),
        }
        for api, path in ((mnc, paths["mojo"]), (nc, paths["reference"])):
            with api.Dataset(path, "w") as dataset:
                dataset.createDimension("sample", n)
                scaled = dataset.createVariable(
                    "scaled", "i2", ("sample",), fill_value=-32767
                )
                scaled.scale_factor = np.float64(0.01)
                scaled.add_offset = np.float64(250.0)
                scaled.valid_range = np.array([-30000, 30000], dtype=np.int16)
                scaled.set_auto_maskandscale(False)
                scaled[:] = raw
                scaled.set_auto_maskandscale(True)

                plain = dataset.createVariable("plain", "f8", ("sample",))
                plain[:] = quantized

                quant = dataset.createVariable(
                    "quantized",
                    "f8",
                    ("sample",),
                    least_significant_digit=3,
                )
                quant[:] = quantized

        mojo_ds = mnc.Dataset(paths["mojo"], "r+")
        ref_ds = nc.Dataset(paths["reference"], "r+")
        try:
            mv = mojo_ds.variables
            rv = ref_ds.variables

            mojo_time = timeit(lambda: mv["scaled"][:])
            ref_time = timeit(lambda: rv["scaled"][:])
            rows.append(
                (
                    "scaled int16 read, 5M",
                    mojo_time,
                    ref_time,
                    result(ref_time, mojo_time),
                )
            )

            mojo_time = timeit(lambda: mv["scaled"].__setitem__(slice(None), physical))
            ref_time = timeit(lambda: rv["scaled"].__setitem__(slice(None), physical))
            rows.append(
                (
                    "scaled int16 write, 5M",
                    mojo_time,
                    ref_time,
                    result(ref_time, mojo_time),
                )
            )

            mojo_time = timeit(lambda: mv["quantized"].__setitem__(slice(None), quantized))
            ref_time = timeit(lambda: rv["quantized"].__setitem__(slice(None), quantized))
            rows.append(
                (
                    "LSD-3 float64 write, 5M",
                    mojo_time,
                    ref_time,
                    result(ref_time, mojo_time),
                )
            )

            mojo_time = timeit(lambda: mv["plain"][:])
            ref_time = timeit(lambda: rv["plain"][:])
            rows.append(
                (
                    "plain float64 read, 5M",
                    mojo_time,
                    ref_time,
                    result(ref_time, mojo_time),
                )
            )

        finally:
            mojo_ds.close()
            ref_ds.close()

    print(
        f"Machine: {cpu_name()}; {platform.system()} {platform.release()}; "
        f"Python {platform.python_version()}; netCDF4 {nc.__version__}"
    )
    print()
    print("| case | mojo-netcdf4 | netCDF4 | result |")
    print("|---|---:|---:|---:|")
    for name, mojo_time, ref_time, comparison in rows:
        print(
            f"| {name} | {mojo_time * 1000:.2f} ms | "
            f"{ref_time * 1000:.2f} ms | {comparison} |"
        )


if __name__ == "__main__":
    main()
