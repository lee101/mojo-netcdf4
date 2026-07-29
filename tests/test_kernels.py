import numpy as np
import pytest

from mojonetcdf4 import _kernels


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

    packed = _kernels.pack(unpacked, "i4", 0.5, 10.0, np.iinfo("i4").max)
    assert np.array_equal(packed, raw)

    values = np.linspace(-10.0, 10.0, n)
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
