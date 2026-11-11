import numpy as np
import pytest

from mojonetcdf4 import _kernels

# The SIMD width the kernels are built with: Float64 lanes on this AVX2 host
# (no AVX-512), which is the `W` in src/netcdf4.mojo for every kernel.
SIMD_W = 4
# Lengths that exercise the vector body and the scalar tail: empty, shorter
# than one vector, every side of one vector, and an odd length well past it.
TAIL_LENGTHS = [0, 1, SIMD_W - 1, SIMD_W, SIMD_W + 1, 100_003]
LARGE_ODD = 1_000_003


@pytest.mark.parametrize(
    "dtype",
    ["i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f4", "f8"],
)
def test_unpack_kernel(dtype):
    raw = np.arange(17, dtype=dtype)
    result, mask = _kernels.unpack(raw, 0.25, -3.0)
    assert mask is None
    assert np.allclose(result, raw.astype(np.float64) * 0.25 - 3.0)


@pytest.mark.parametrize("dtype", [">i2", ">u4", ">f8"])
def test_unpack_and_mask_accept_non_native_endian_input(dtype):
    raw = np.array([1, 2, 99, 4, 5], dtype=dtype)
    result, mask = _kernels.unpack(raw, 0.5, -1.0, fill=99)
    assert np.array_equal(result, [ -0.5, 0.0, 48.5, 1.0, 1.5])
    assert np.array_equal(mask, [False, False, True, False, False])


@pytest.mark.parametrize("dtype", ["i1", "u1", "i2", "u2", "i4", "u4"])
def test_pack_kernel(dtype):
    info = np.iinfo(dtype)
    raw = np.arange(17, dtype=np.float64)
    data = raw * 0.5 + 10.0
    result = _kernels.pack(data, dtype, 0.5, 10.0, info.max)
    assert np.array_equal(result, raw.astype(dtype))


def test_mask_kernel_simd_tail():
    raw = np.array(
        [-99, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99],
        dtype=np.int16,
    )
    mask = _kernels.make_mask(
        raw,
        fill=np.int16(-99),
        missing=np.int16(5),
        valid_min=np.int16(-2),
        valid_max=np.int16(10),
    )
    assert np.array_equal(
        mask,
        [True, True, False, False, False, False, False, False, False,
         True, False, False, False, False, False, True, True],
    )


def test_mask_kernel_and_exact_int64_fallback():
    raw = np.array(
        [-9223372036854775806, -2, 0, 4, 9223372036854775806],
        dtype=np.int64,
    )
    mask = _kernels.make_mask(
        raw,
        fill=np.int64(-9223372036854775806),
        missing=np.int64(4),
        valid_max=np.int64(9223372036854775805),
    )
    assert np.array_equal(mask, [True, False, False, True, True])


def test_quantize_kernel_matches_netcdf_formula():
    values = np.linspace(-10, 10, 1001)
    result = _kernels.quantize(values, 2)
    multiplier = 2.0 ** np.ceil(np.log2(100.0))
    assert np.array_equal(result, np.around(multiplier * values) / multiplier)


def test_parallel_threshold_paths():
    n = _kernels.PARALLEL_THRESHOLD + 3
    raw = np.arange(n, dtype=np.int32)
    decoded, no_mask = _kernels.unpack(raw, 0.5, 10.0)
    assert no_mask is None
    assert np.array_equal(decoded, raw.astype(np.float64) * 0.5 + 10.0)

    unpacked, mask = _kernels.unpack(
        raw,
        0.5,
        10.0,
        fill=np.int32(17),
        valid_max=np.int32(n - 2),
    )
    assert np.array_equal(unpacked, raw.astype(np.float64) * 0.5 + 10.0)
    assert np.count_nonzero(mask) == 2
    assert mask[17] and mask[-1]
    direct_mask = _kernels.make_mask(
        raw, fill=np.int32(17), valid_max=np.int32(n - 2)
    )
    assert np.array_equal(direct_mask, mask)

    packed = _kernels.pack(unpacked, "i4", 0.5, 10.0, np.iinfo("i4").max)
    assert np.array_equal(packed, raw)

    values = np.linspace(-10.0, 10.0, n)
    quantized = _kernels.quantize(values, 3)
    multiplier = 2.0 ** np.ceil(np.log2(1000.0))
    assert np.array_equal(
        quantized, np.around(multiplier * values) / multiplier
    )


def test_serial_parallel_threshold_boundary():
    for n in (_kernels.PARALLEL_THRESHOLD - 1, _kernels.PARALLEL_THRESHOLD):
        raw = np.arange(n, dtype=np.int16)
        unpacked, mask = _kernels.unpack(
            raw, 0.25, -3.0, fill=np.int16(-1)
        )
        assert np.array_equal(unpacked, raw.astype(np.float64) * 0.25 - 3.0)
        assert mask.sum() == n // 65536

        values = np.linspace(-1.0, 1.0, n)
        quantized = _kernels.quantize(values, 3)
        multiplier = 2.0 ** np.ceil(np.log2(1000.0))
        assert np.array_equal(
            quantized, np.around(multiplier * values) / multiplier
        )


def test_kernel_errors_are_not_silently_ignored():
    with pytest.raises(RuntimeError, match="rejected its arguments"):
        _kernels._checked_call(
            "mnc_quantize_f64", 0, 0, 1, 1.0
        )


# --- SIMD tail coverage -----------------------------------------------------
# Every kernel runs a W-lane body and a scalar tail. These cases pin the four
# fused/masked kernels at lengths on both sides of W, at 0 and 1, and at a long
# odd length, and compare against a NumPy reference of the same formula.

LENGTHS = TAIL_LENGTHS + [LARGE_ODD]


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize(
    "dtype", ["i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f4", "f8"]
)
def test_unpack_with_mask_simd_tail(n, dtype):
    dtype = np.dtype(dtype)
    info = np.iinfo(dtype) if dtype.kind in "iu" else None
    top = float(info.max) if info is not None else float(n)
    raw = np.arange(n, dtype=dtype) if info is None else np.minimum(
        np.arange(n), top
    ).astype(dtype)
    fill = dtype.type(3)
    valid_min = dtype.type(2)
    valid_max = dtype.type(min(max(n - 2, 0), int(top)))
    decoded, mask = _kernels.unpack(
        raw,
        0.5,
        -1.0,
        fill=fill,
        valid_min=valid_min,
        valid_max=valid_max,
    )
    assert np.array_equal(
        decoded, raw.astype(np.float64) * 0.5 - 1.0
    )
    expected = (raw == fill) | (raw < valid_min) | (raw > valid_max)
    assert np.array_equal(mask, expected)
    # The mask kernel alone must agree with the mask the fused decode produced.
    assert np.array_equal(
        _kernels.make_mask(
            raw, fill=fill, valid_min=valid_min, valid_max=valid_max
        ),
        expected,
    )


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("dtype", ["i1", "u1", "i2", "u2", "f4", "f8"])
def test_pack_simd_tail(n, dtype):
    info = np.iinfo(np.dtype(dtype)) if np.dtype(dtype).kind in "iu" else None
    fill = info.max if info is not None else np.dtype(dtype).type(-32767.0)
    values = np.arange(n, dtype=np.float64) - n / 2.0
    data = np.ma.array(
        values,
        mask=np.arange(n) % 7 == 0,
    )
    packed = _kernels.pack(data, dtype, 0.5, 10.0, fill)
    unmasked = ~np.ma.getmaskarray(data)
    expected = np.full(n, fill, dtype=np.dtype(dtype))
    expected[unmasked] = (values[unmasked] - 10.0) / 0.5
    assert np.array_equal(packed, expected.astype(np.dtype(dtype)))


@pytest.mark.parametrize("n", LENGTHS)
def test_quantize_simd_tail(n):
    values = np.linspace(-1.0, 1.0, n)
    quantized = _kernels.quantize(values, 3)
    multiplier = 2.0 ** np.ceil(np.log2(1000.0))
    assert np.array_equal(
        quantized, np.around(multiplier * values) / multiplier
    )


# --- Serial vs parallel: identical output on identical input ----------------
# These are the only tests that take the worker-pool path (the threshold is
# 8,388,608 elements). They build the same source with PARALLEL_THRESHOLD set
# past every test length, so the *library itself* dispatches to the serial body
# while the normal build dispatches to `parallelize`, and compare the two
# outputs lane by lane on the same input.

import os  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402

SERIAL_ONLY = None


@pytest.fixture(scope="module")
def serial_lib():
    """A second build of the same source whose kernels never parallelize."""
    global SERIAL_ONLY
    if SERIAL_ONLY is not None:
        return SERIAL_ONLY
    with open(_kernels.SRC, encoding="utf-8") as stream:
        source = stream.read()
    needle = "comptime PARALLEL_THRESHOLD = 10485760"
    if needle not in source:
        pytest.skip("parallel threshold constant not found in the kernel source")
    source = source.replace(needle, "comptime PARALLEL_THRESHOLD = 9223372036854775807")
    tmp = tempfile.mkdtemp(prefix="mnc-serial-")
    src_path = os.path.join(tmp, "netcdf4.mojo")
    out_path = os.path.join(tmp, "libmojo-netcdf4-serial.so")
    with open(src_path, "w", encoding="utf-8") as stream:
        stream.write(source)
    proc = subprocess.run(
        ["bash", os.path.join(_kernels.ROOT, "build", "build.sh")],
        env=dict(os.environ, MNC_SRC=src_path, MNC_OUT=out_path),
        capture_output=True,
        text=True,
    )
    if proc.returncode or not os.path.exists(out_path):
        pytest.skip(f"serial reference build failed: {proc.stderr[-2000:]}")
    SERIAL_ONLY = out_path
    return SERIAL_ONLY


def _loaded(path):
    import ctypes

    lib = ctypes.CDLL(path)
    for name, (argtypes, restype) in _kernels._SIGNATURES.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
    return lib


def _inputs(n):
    """Same bytes for both builds: pseudo-random packed i2 with a mask."""
    rng = np.random.default_rng(2026)
    raw = np.rint(rng.normal(280.0, 12.0, n)).astype(np.int16)
    raw[: n // 2000] = -32767
    raw[-max(n // 3000, 1):] = 25000
    return raw


def test_serial_and_parallel_outputs_match(serial_lib):
    import ctypes

    n = _kernels.PARALLEL_THRESHOLD + 1_000_003
    raw = _inputs(n)
    raw64 = raw.astype(np.float64)
    packed_theirs = np.full(n, 0, dtype=np.int16)

    par = _kernels.lib()
    ser = _loaded(serial_lib)

    # unpack with the four mask kinds live (fill + valid_min + valid_max).
    decoded = np.empty(n, dtype=np.float64)
    mask = np.empty(n, dtype=np.uint8)
    decoded_ser = np.empty(n, dtype=np.float64)
    mask_ser = np.empty(n, dtype=np.uint8)
    args = (n, 3, 0.01, 250.0, -32767.0, 0.0, -30000.0, 30000.0, 1 | 4 | 8)
    assert par.mnc_unpack_f64(
        raw.ctypes.data, decoded.ctypes.data, raw.ctypes.data,
        mask.ctypes.data, *args,
    ) == 0
    assert ser.mnc_unpack_f64(
        raw.ctypes.data, decoded_ser.ctypes.data, raw.ctypes.data,
        mask_ser.ctypes.data, *args,
    ) == 0
    assert decoded.tobytes() == decoded_ser.tobytes()
    assert mask.tobytes() == mask_ser.tobytes()

    # decode only (no mask buffer).
    decoded[:] = np.nan
    decoded_ser[:] = np.nan
    assert par.mnc_unpack_f64(
        raw.ctypes.data, decoded.ctypes.data, 0, 0,
        n, 3, 0.01, 250.0, 0.0, 0.0, 0.0, 0.0, 0,
    ) == 0
    assert ser.mnc_unpack_f64(
        raw.ctypes.data, decoded_ser.ctypes.data, 0, 0,
        n, 3, 0.01, 250.0, 0.0, 0.0, 0.0, 0.0, 0,
    ) == 0
    assert decoded.tobytes() == decoded_ser.tobytes()

    # mask kernel alone.
    mask[:] = 7
    mask_ser[:] = 7
    mask_args = (n, 3, -32767.0, 0.0, -30000.0, 30000.0, 1 | 4 | 8)
    assert par.mnc_mask_f64(
        raw.ctypes.data, mask.ctypes.data, *mask_args,
    ) == 0
    assert ser.mnc_mask_f64(
        raw.ctypes.data, mask_ser.ctypes.data, *mask_args,
    ) == 0
    assert mask.tobytes() == mask_ser.tobytes()

    # masked pack, integer storage.
    source_mask = (np.arange(n) % 977 == 0).astype(np.uint8)
    packed = np.full(n, 0, dtype=np.int16)
    packed_theirs[:] = -1
    assert par.mnc_pack_f64(
        raw64.ctypes.data, source_mask.ctypes.data, packed.ctypes.data,
        n, 3, 0.01, 250.0, -32767.0,
    ) == 0
    assert ser.mnc_pack_f64(
        raw64.ctypes.data, source_mask.ctypes.data, packed_theirs.ctypes.data,
        n, 3, 0.01, 250.0, -32767.0,
    ) == 0
    assert packed.tobytes() == packed_theirs.tobytes()

    # quantization.
    quantized = np.empty(n, dtype=np.float64)
    quantized_ser = np.empty(n, dtype=np.float64)
    assert par.mnc_quantize_f64(
        raw64.ctypes.data, quantized.ctypes.data, n, 1024.0,
    ) == 0
    assert ser.mnc_quantize_f64(
        raw64.ctypes.data, quantized_ser.ctypes.data, n, 1024.0,
    ) == 0
    assert quantized.tobytes() == quantized_ser.tobytes()


def test_public_path_serial_and_parallel_agree(serial_lib):
    """The public wrappers, not the raw symbols, on the same arrays."""
    n = _kernels.PARALLEL_THRESHOLD + 3
    raw = _inputs(n)
    source_mask = np.arange(n) % 3 == 0
    kwargs = dict(fill=np.int16(-32767), valid_min=np.int16(-30000))

    # Swap the module's library for the serial-only build, then restore.
    original = _kernels._LIB
    try:
        _kernels._LIB = _loaded(serial_lib)
        seq, mask_seq = _kernels.unpack(raw, 0.01, 250.0, **kwargs)
        seq_mask = _kernels.make_mask(raw, **kwargs)
        seq_packed = _kernels.pack(
            np.ma.array(raw, mask=source_mask),
            "i2", 0.01, 250.0, np.int16(-32767),
        )
        seq_quant = _kernels.quantize(raw.astype(np.float64), 3)
    finally:
        _kernels._LIB = original

    par, mask_par = _kernels.unpack(raw, 0.01, 250.0, **kwargs)
    par_mask = _kernels.make_mask(raw, **kwargs)
    packed = _kernels.pack(
        np.ma.array(raw, mask=source_mask),
        "i2", 0.01, 250.0, np.int16(-32767),
    )
    quant = _kernels.quantize(raw.astype(np.float64), 3)

    assert par.tobytes() == seq.tobytes()
    assert mask_par.tobytes() == mask_seq.tobytes()
    assert par_mask.tobytes() == seq_mask.tobytes()
    assert packed.tobytes() == seq_packed.tobytes()
    assert quant.tobytes() == seq_quant.tobytes()


