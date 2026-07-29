"""ctypes bridge to the Mojo netCDF numeric kernels."""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "netcdf4.mojo")
LIB = os.path.join(ROOT, "dist", "libmojo-netcdf4.so")

I = ctypes.c_int64
F = ctypes.c_double
_SIGNATURES = {
    "mnc_unpack_f64": ([I, I, I, I, I, I, F, F, F, F, F, F, I], I),
    "mnc_mask_f64": ([I, I, I, I, F, F, F, F, I], I),
    "mnc_pack_f64": ([I, I, I, I, I, F, F, F], I),
    "mnc_quantize_f64": ([I, I, I, F], I),
}
PARALLEL_THRESHOLD = 1_048_576

_KINDS = {
    np.dtype("i1"): 1,
    np.dtype("u1"): 2,
    np.dtype("i2"): 3,
    np.dtype("u2"): 4,
    np.dtype("i4"): 5,
    np.dtype("u4"): 6,
    np.dtype("i8"): 7,
    np.dtype("u8"): 8,
    np.dtype("f4"): 9,
    np.dtype("f8"): 10,
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SRC)
    ):
        return LIB
    if not shutil.which("mojo"):
        raise BuildError("mojo not found; run inside `pixi run`")
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIB = None


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_LIB, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _LIB


def kind_for(dtype) -> int:
    dtype = np.dtype(dtype).newbyteorder("=")
    try:
        return _KINDS[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported netCDF numeric dtype {dtype}") from exc


def _native_contiguous(array) -> np.ndarray:
    """Return C-contiguous native-endian storage suitable for the C ABI."""
    array = np.asarray(array)
    dtype = array.dtype.newbyteorder("=")
    return np.ascontiguousarray(array, dtype=dtype)


def _checked_call(name: str, *args) -> None:
    status = getattr(lib(), name)(*args)
    if status != 0:
        raise RuntimeError(f"Mojo kernel {name} rejected its arguments")


def _mask_numpy(
    raw: np.ndarray,
    fill,
    missing,
    valid_min,
    valid_max,
) -> np.ndarray:
    mask = np.zeros(raw.shape, dtype=bool)
    for value in np.asarray(missing).reshape(-1) if missing is not None else ():
        mask |= np.isnan(raw) if np.issubdtype(raw.dtype, np.floating) and np.isnan(value) else raw == value
    if fill is not None:
        mask |= np.isnan(raw) if np.issubdtype(raw.dtype, np.floating) and np.isnan(fill) else raw == fill
    if valid_min is not None:
        mask |= raw < valid_min
    if valid_max is not None:
        mask |= raw > valid_max
    return mask


def make_mask(
    raw: np.ndarray,
    *,
    fill=None,
    missing=None,
    valid_min=None,
    valid_max=None,
) -> np.ndarray:
    shape = np.shape(raw)
    raw = _native_contiguous(raw)
    if not raw.size:
        return np.zeros(shape, dtype=bool)
    missing_values = (
        np.asarray(missing).reshape(-1) if missing is not None else np.empty(0)
    )
    if raw.dtype.kind in "iu" and raw.dtype.itemsize == 8:
        return _mask_numpy(
            raw, fill, missing_values, valid_min, valid_max
        ).reshape(shape)

    mask = np.empty(raw.size, dtype=np.uint8)
    flags = 0
    fill64 = missing64 = min64 = max64 = 0.0
    if fill is not None:
        flags |= 1
        fill64 = float(fill)
    if missing_values.size:
        flags |= 2
        missing64 = float(missing_values[0])
    if valid_min is not None:
        flags |= 4
        min64 = float(valid_min)
    if valid_max is not None:
        flags |= 8
        max64 = float(valid_max)
    _checked_call(
        "mnc_mask_f64",
        raw.ctypes.data,
        mask.ctypes.data,
        raw.size,
        kind_for(raw.dtype),
        fill64,
        missing64,
        min64,
        max64,
        flags,
    )
    result = mask.view(bool).reshape(raw.shape)
    if missing_values.size > 1:
        for value in missing_values[1:]:
            result |= (
                np.isnan(raw)
                if np.issubdtype(raw.dtype, np.floating) and np.isnan(value)
                else raw == value
            )
    return result.reshape(shape)


def unpack(
    raw: np.ndarray,
    scale: float,
    offset: float,
    *,
    fill=None,
    missing=None,
    valid_min=None,
    valid_max=None,
) -> tuple[np.ndarray, np.ndarray | None]:
    shape = np.shape(raw)
    raw = _native_contiguous(raw)
    result = np.empty(raw.size, dtype=np.float64)
    mask = None
    mask_addr = flags = 0
    fill64 = missing64 = min64 = max64 = 0.0
    missing_values = (
        np.asarray(missing).reshape(-1) if missing is not None else np.empty(0)
    )
    has_mask_config = any(
        x is not None for x in (fill, missing, valid_min, valid_max)
    )
    exact_mask = raw.dtype.kind in "iu" and raw.dtype.itemsize == 8
    if has_mask_config:
        if exact_mask:
            mask = make_mask(
                raw,
                fill=fill,
                missing=missing_values,
                valid_min=valid_min,
                valid_max=valid_max,
            )
        else:
            mask = np.empty(raw.size, dtype=np.uint8)
            mask_addr = mask.ctypes.data
            if fill is not None:
                flags |= 1
                fill64 = float(fill)
            if missing_values.size:
                flags |= 2
                missing64 = float(missing_values[0])
            if valid_min is not None:
                flags |= 4
                min64 = float(valid_min)
            if valid_max is not None:
                flags |= 8
                max64 = float(valid_max)
    if raw.size:
        _checked_call(
            "mnc_unpack_f64",
            raw.ctypes.data,
            result.ctypes.data,
            raw.ctypes.data if mask_addr else 0,
            mask_addr,
            raw.size,
            kind_for(raw.dtype),
            float(scale),
            float(offset),
            fill64,
            missing64,
            min64,
            max64,
            flags,
        )
    if mask_addr:
        mask = mask.view(bool).reshape(raw.shape)
        if missing_values.size > 1:
            for value in missing_values[1:]:
                mask |= (
                    np.isnan(raw)
                    if np.issubdtype(raw.dtype, np.floating) and np.isnan(value)
                    else raw == value
                )
    return result.reshape(shape), None if mask is None else mask.reshape(shape)


def pack(
    data,
    dtype,
    scale: float,
    offset: float,
    fill,
) -> np.ndarray:
    shape = np.shape(np.ma.getdata(data))
    values = np.ascontiguousarray(np.ma.getdata(data), dtype=np.float64)
    result = np.empty(values.shape, dtype=np.dtype(dtype).newbyteorder("="))
    if scale == 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            packed = np.around((values - offset) / scale)
        if np.ma.isMaskedArray(data):
            packed[np.ma.getmaskarray(data)] = fill
        return packed.astype(result.dtype).reshape(shape)
    mask_addr = 0
    if np.ma.isMaskedArray(data):
        mask = np.ascontiguousarray(np.ma.getmaskarray(data), dtype=np.uint8)
        if mask.any():
            mask_addr = mask.ctypes.data
    if values.size:
        _checked_call(
            "mnc_pack_f64",
            values.ctypes.data,
            mask_addr,
            result.ctypes.data,
            values.size,
            kind_for(result.dtype),
            float(scale),
            float(offset),
            float(fill),
        )
    return result.reshape(shape)


def quantize(data, least_significant_digit: int):
    precision = 10.0 ** -least_significant_digit
    exponent = math.log10(precision)
    exponent = math.floor(exponent) if exponent < 0 else math.ceil(exponent)
    multiplier = 2.0 ** math.ceil(math.log2(10.0 ** -exponent))
    shape = np.shape(np.ma.getdata(data))
    values = np.ascontiguousarray(np.ma.getdata(data), dtype=np.float64)
    result = np.empty_like(values)
    if values.size:
        _checked_call(
            "mnc_quantize_f64",
            values.ctypes.data, result.ctypes.data, values.size, multiplier
        )
    if np.ma.isMaskedArray(data):
        result = np.ma.array(result, mask=np.ma.getmaskarray(data), copy=False)
        result.set_fill_value(data.fill_value)
    return result.reshape(shape)
