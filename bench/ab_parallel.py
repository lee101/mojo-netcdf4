"""Serial vs parallel A/B for the four kernels, interleaved in one process.

Both libraries must be builds of the same source that differ only in the
switch under test:

  serial_lib  PARALLEL_THRESHOLD set past every test length (always serial)
  par_lib     the build under test

Build the serial reference from MNC_SRC/MNC_OUT without touching dist/:

  python - <<'EOF'
  s = open("src/netcdf4.mojo").read()
  open("/tmp/serial_only.mojo", "w").write(s.replace(
      "comptime PARALLEL_THRESHOLD = 10485760",
      "comptime PARALLEL_THRESHOLD = 9223372036854775807"))
  EOF
  MNC_SRC=/tmp/serial_only.mojo MNC_OUT=/tmp/lib_serial_only.so \
      bash build/build.sh
  pixi run build          # restores dist/libmojo-netcdf4.so

Then, under the bench lock:

  flock /tmp/mojo-bench-par.lock pixi run python bench/ab_parallel.py \
      /tmp/lib_serial_only.so dist/libmojo-netcdf4.so \
      1048576,4194304,8388608,10485759,10485760,20971520 13

Times are best-of-N per library, alternating, so the two builds see the same
host load. "speedup" is serial/parallel; < 1.00x means the parallel path lost.
"""

from __future__ import annotations

import ctypes
import sys
import time

import numpy as np

I = ctypes.c_int64
F = ctypes.c_double
SIGNATURES = {
    "mnc_unpack_f64": [I, I, I, I, I, I, F, F, F, F, F, F, I],
    "mnc_mask_f64": [I, I, I, I, F, F, F, F, I],
    "mnc_pack_f64": [I, I, I, I, I, F, F, F],
    "mnc_quantize_f64": [I, I, I, F],
}


def load(path):
    lib = ctypes.CDLL(path)
    for name, args in SIGNATURES.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = I
    return lib


def best(call, repeat):
    call()
    fastest = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        call()
        finest = time.perf_counter() - start
        if finest < fastest:
            fastest = finest
    return fastest * 1e3


def main():
    serial_path, parallel_path, sizes, repeat = (
        sys.argv[1],
        sys.argv[2],
        [int(x) for x in sys.argv[3].split(",")],
        int(sys.argv[4]),
    )
    serial = load(serial_path)
    parallel = load(parallel_path)
    rng = np.random.default_rng(11)
    for n in sizes:
        raw = np.rint(rng.normal(280.0, 12.0, n)).astype(np.int16)
        raw[: max(n // 2000, 1)] = -32767  # ~0.05 % fill values
        raw64 = raw.astype(np.float64)
        dst = np.empty(n, dtype=np.float64)
        mask = np.empty(n, dtype=np.uint8)
        packed = np.empty(n, dtype=np.int16)
        calls = {
            "unpack": lambda lib: lib.mnc_unpack_f64(
                raw.ctypes.data, dst.ctypes.data, raw.ctypes.data,
                mask.ctypes.data, n, 3, 0.01, 250.0, -32767.0, 0.0,
                -30000.0, 30000.0, 1 | 4 | 8,
            ),
            "unpack_nomask": lambda lib: lib.mnc_unpack_f64(
                raw.ctypes.data, dst.ctypes.data, 0, 0, n, 3, 0.01, 250.0,
                0.0, 0.0, 0.0, 0.0, 0,
            ),
            "mask": lambda lib: lib.mnc_mask_f64(
                raw.ctypes.data, mask.ctypes.data, n, 3, -32767.0, 0.0,
                -30000.0, 30000.0, 1 | 4 | 8,
            ),
            "pack": lambda lib: lib.mnc_pack_f64(
                raw64.ctypes.data, mask.ctypes.data, packed.ctypes.data,
                n, 3, 0.01, 250.0, -32767.0,
            ),
            "quantize": lambda lib: lib.mnc_quantize_f64(
                dst.ctypes.data, dst.ctypes.data, n, 1024.0,
            ),
        }
        cells = []
        for name, call in calls.items():
            serial_ms = best(lambda: call(serial), repeat)
            parallel_ms = best(lambda: call(parallel), repeat)
            cells.append(
                f"{name}: {serial_ms:7.2f}->{parallel_ms:7.2f} "
                f"({serial_ms / parallel_ms:4.2f}x)"
            )
        print(f"{n:>9} " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
