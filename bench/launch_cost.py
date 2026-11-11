"""Where the toolchain's parallel launch cost appears.

Builds of the same source at several GRAIN_SIZE values and worker caps are
loaded side by side and one kernel is timed across chunk counts. The numbers
this prints are the ones quoted in bench/analysis.md §12:

  * the ~5 ms floor appears at a total element count, not a chunk count,
  * the cost is per `parallelize` call (two back-to-back launches each pay it),
  * so the threshold, not the grain, is what protects small inputs.

Usage (under the bench lock, from the repo root):

  pixi run python bench/launch_cost.py /tmp/lib_g262144.so 262144 1,2,3,4,8,16

arg 1 is a library built with the given grain and a cap >= 4; arg 2 is that
grain; arg 3 is the list of chunk counts to time.
"""

from __future__ import annotations

import ctypes
import sys
import time

import numpy as np

I = ctypes.c_int64
F = ctypes.c_double


def main():
    path, grain, chunk_counts = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    lib = ctypes.CDLL(path)
    fn = lib.mnc_quantize_f64
    fn.argtypes = [I, I, I, F]
    fn.restype = I
    buffer = np.ones(64_000_000)  # room for 16 x 1 MiB-grain chunks
    print(f"quantize, grain {grain}, {path}")
    print(f"{'chunks':>7} {'n':>10} {'ms':>9} {'ms/M':>8}")
    for chunks in [int(x) for x in chunk_counts.split(",")]:
        n = grain * chunks
        call = lambda: fn(  # noqa: E731
            buffer.ctypes.data, buffer.ctypes.data, n, 1024.0
        )
        call()
        fastest = float("inf")
        for _ in range(21):
            start = time.perf_counter()
            call()
            finest = time.perf_counter() - start
            if finest < fastest:
                fastest = finest
        ms = fastest * 1e3
        print(f"{chunks:>7} {n:>10} {ms:9.3f} {ms / (n / 1e6):8.3f}")


if __name__ == "__main__":
    main()
