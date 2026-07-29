"""Numeric transforms around netCDF variable I/O."""

from std.algorithm import parallelize
from std.gpu.host import DeviceContext
from std.math import round
from std.sys.info import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime DPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime PARALLEL_THRESHOLD = 1_048_576
comptime PARALLEL_GRAIN = 262_144


@always_inline
def masked_value(
    value: Float64,
    fill: Float64,
    missing: Float64,
    valid_min: Float64,
    valid_max: Float64,
    flags: Int,
) -> Bool:
    if (flags & 1) != 0:
        if (fill != fill and value != value) or value == fill:
            return True
    if (flags & 2) != 0:
        if (missing != missing and value != value) or value == missing:
            return True
    if (flags & 4) != 0 and value < valid_min:
        return True
    if (flags & 8) != 0 and value > valid_max:
        return True
    return False


def decode_range(
    src: BPtr,
    dst: DPtr,
    start: Int,
    end: Int,
    kind: Int,
    scale: Float64,
    offset: Float64,
):
    var i = start
    if kind == 1:
        var values = src.bitcast[Int8]()
        while i + W <= end:
            dst.store(
                i,
                values.load[width=W, alignment=1](i).cast[DType.float64]()
                * scale
                + offset,
            )
            i += W
        while i < end:
            dst[i] = Float64(values[i]) * scale + offset
            i += 1
    elif kind == 2:
        var values = src.bitcast[UInt8]()
        while i + W <= end:
            dst.store(
                i,
                values.load[width=W, alignment=1](i).cast[DType.float64]()
                * scale
                + offset,
            )
            i += W
        while i < end:
            dst[i] = Float64(values[i]) * scale + offset
            i += 1
    elif kind == 3:
        var values = src.bitcast[Int16]()
        while i + W <= end:
            dst.store(
                i,
                values.load[width=W, alignment=1](i).cast[DType.float64]()
                * scale
                + offset,
            )
            i += W
        while i < end:
            dst[i] = Float64(values[i]) * scale + offset
            i += 1
    elif kind == 4:
        var values = src.bitcast[UInt16]()
        while i + W <= end:
            dst.store(
                i,
                values.load[width=W, alignment=1](i).cast[DType.float64]()
                * scale
                + offset,
            )
            i += W
        while i < end:
            dst[i] = Float64(values[i]) * scale + offset
            i += 1
    elif kind == 5:
        var values = src.bitcast[Int32]()
        while i + W <= end:
            dst.store(
                i,
                values.load[width=W, alignment=1](i).cast[DType.float64]()
                * scale
                + offset,
            )
            i += W
        while i < end:
            dst[i] = Float64(values[i]) * scale + offset
            i += 1
    elif kind == 6:
        var values = src.bitcast[UInt32]()
        while i + W <= end:
            dst.store(
                i,
                values.load[width=W, alignment=1](i).cast[DType.float64]()
                * scale
                + offset,
            )
            i += W
        while i < end:
            dst[i] = Float64(values[i]) * scale + offset
            i += 1
    elif kind == 7:
        var values = src.bitcast[Int64]()
        while i + W <= end:
            dst.store(
                i,
                values.load[width=W, alignment=1](i).cast[DType.float64]()
                * scale
                + offset,
            )
            i += W
        while i < end:
            dst[i] = Float64(values[i]) * scale + offset
            i += 1
    elif kind == 8:
        var values = src.bitcast[UInt64]()
        while i + W <= end:
            dst.store(
                i,
                values.load[width=W, alignment=1](i).cast[DType.float64]()
                * scale
                + offset,
            )
            i += W
        while i < end:
            dst[i] = Float64(values[i]) * scale + offset
            i += 1
    elif kind == 9:
        var values = src.bitcast[Float32]()
        while i + W <= end:
            dst.store(
                i,
                values.load[width=W, alignment=1](i).cast[DType.float64]()
                * scale
                + offset,
            )
            i += W
        while i < end:
            dst[i] = Float64(values[i]) * scale + offset
            i += 1
    else:
        var values = src.bitcast[Float64]()
        while i + W <= end:
            dst.store(
                i, values.load[width=W, alignment=1](i) * scale + offset
            )
            i += W
        while i < end:
            dst[i] = values[i] * scale + offset
            i += 1


def decode_values(
    src: BPtr,
    dst: DPtr,
    n: Int,
    kind: Int,
    scale: Float64,
    offset: Float64,
):
    if n < PARALLEL_THRESHOLD:
        decode_range(src, dst, 0, n, kind, scale, offset)
        return

    var tasks = (n + PARALLEL_GRAIN - 1) // PARALLEL_GRAIN

    @parameter
    @__copy_capture(src, dst, n, kind, scale, offset)
    def work(task: Int):
        var start = task * PARALLEL_GRAIN
        var end = min(start + PARALLEL_GRAIN, n)
        decode_range(src, dst, start, end, kind, scale, offset)

    try:
        var ctx = DeviceContext(api="cpu")
        parallelize[work](tasks, ctx=ctx)
    except:
        decode_range(src, dst, 0, n, kind, scale, offset)


@always_inline
def store_mask_vector(
    mask: BPtr,
    i: Int,
    values: SIMD[DType.float64, W],
    fill: Float64,
    missing: Float64,
    valid_min: Float64,
    valid_max: Float64,
    flags: Int,
):
    var masked = SIMD[DType.bool, W](fill=False)
    if (flags & 1) != 0:
        if fill != fill:
            masked = masked | values.ne(values)
        else:
            masked = masked | values.eq(fill)
    if (flags & 2) != 0:
        if missing != missing:
            masked = masked | values.ne(values)
        else:
            masked = masked | values.eq(missing)
    if (flags & 4) != 0:
        masked = masked | values.lt(valid_min)
    if (flags & 8) != 0:
        masked = masked | values.gt(valid_max)
    mask.store(
        i,
        masked.select(
            SIMD[DType.uint8, W](1),
            SIMD[DType.uint8, W](0),
        ),
    )


@always_inline
def store_mask_scalar(
    mask: BPtr,
    i: Int,
    value: Float64,
    fill: Float64,
    missing: Float64,
    valid_min: Float64,
    valid_max: Float64,
    flags: Int,
):
    mask[i] = UInt8(
        1
        if masked_value(
            value, fill, missing, valid_min, valid_max, flags
        )
        else 0
    )


def mask_range(
    src: BPtr,
    mask: BPtr,
    start: Int,
    end: Int,
    kind: Int,
    fill: Float64,
    missing: Float64,
    valid_min: Float64,
    valid_max: Float64,
    flags: Int,
):
    var i = start
    if kind == 1:
        var values = src.bitcast[Int8]()
        while i + W <= end:
            store_mask_vector(
                mask, i, values.load[width=W, alignment=1](i).cast[DType.float64](),
                fill, missing, valid_min, valid_max, flags,
            )
            i += W
        while i < end:
            store_mask_scalar(
                mask, i, Float64(values[i]), fill, missing,
                valid_min, valid_max, flags,
            )
            i += 1
    elif kind == 2:
        var values = src.bitcast[UInt8]()
        while i + W <= end:
            store_mask_vector(
                mask, i, values.load[width=W, alignment=1](i).cast[DType.float64](),
                fill, missing, valid_min, valid_max, flags,
            )
            i += W
        while i < end:
            store_mask_scalar(
                mask, i, Float64(values[i]), fill, missing,
                valid_min, valid_max, flags,
            )
            i += 1
    elif kind == 3:
        var values = src.bitcast[Int16]()
        while i + W <= end:
            store_mask_vector(
                mask, i, values.load[width=W, alignment=1](i).cast[DType.float64](),
                fill, missing, valid_min, valid_max, flags,
            )
            i += W
        while i < end:
            store_mask_scalar(
                mask, i, Float64(values[i]), fill, missing,
                valid_min, valid_max, flags,
            )
            i += 1
    elif kind == 4:
        var values = src.bitcast[UInt16]()
        while i + W <= end:
            store_mask_vector(
                mask, i, values.load[width=W, alignment=1](i).cast[DType.float64](),
                fill, missing, valid_min, valid_max, flags,
            )
            i += W
        while i < end:
            store_mask_scalar(
                mask, i, Float64(values[i]), fill, missing,
                valid_min, valid_max, flags,
            )
            i += 1
    elif kind == 5:
        var values = src.bitcast[Int32]()
        while i + W <= end:
            store_mask_vector(
                mask, i, values.load[width=W, alignment=1](i).cast[DType.float64](),
                fill, missing, valid_min, valid_max, flags,
            )
            i += W
        while i < end:
            store_mask_scalar(
                mask, i, Float64(values[i]), fill, missing,
                valid_min, valid_max, flags,
            )
            i += 1
    elif kind == 6:
        var values = src.bitcast[UInt32]()
        while i + W <= end:
            store_mask_vector(
                mask, i, values.load[width=W, alignment=1](i).cast[DType.float64](),
                fill, missing, valid_min, valid_max, flags,
            )
            i += W
        while i < end:
            store_mask_scalar(
                mask, i, Float64(values[i]), fill, missing,
                valid_min, valid_max, flags,
            )
            i += 1
    elif kind == 7:
        var values = src.bitcast[Int64]()
        while i + W <= end:
            store_mask_vector(
                mask, i, values.load[width=W, alignment=1](i).cast[DType.float64](),
                fill, missing, valid_min, valid_max, flags,
            )
            i += W
        while i < end:
            store_mask_scalar(
                mask, i, Float64(values[i]), fill, missing,
                valid_min, valid_max, flags,
            )
            i += 1
    elif kind == 8:
        var values = src.bitcast[UInt64]()
        while i + W <= end:
            store_mask_vector(
                mask, i, values.load[width=W, alignment=1](i).cast[DType.float64](),
                fill, missing, valid_min, valid_max, flags,
            )
            i += W
        while i < end:
            store_mask_scalar(
                mask, i, Float64(values[i]), fill, missing,
                valid_min, valid_max, flags,
            )
            i += 1
    elif kind == 9:
        var values = src.bitcast[Float32]()
        while i + W <= end:
            store_mask_vector(
                mask, i, values.load[width=W, alignment=1](i).cast[DType.float64](),
                fill, missing, valid_min, valid_max, flags,
            )
            i += W
        while i < end:
            store_mask_scalar(
                mask, i, Float64(values[i]), fill, missing,
                valid_min, valid_max, flags,
            )
            i += 1
    else:
        var values = src.bitcast[Float64]()
        while i + W <= end:
            store_mask_vector(
                mask, i, values.load[width=W, alignment=1](i),
                fill, missing, valid_min, valid_max, flags,
            )
            i += W
        while i < end:
            store_mask_scalar(
                mask, i, values[i], fill, missing,
                valid_min, valid_max, flags,
            )
            i += 1


def mask_values(
    src: BPtr,
    mask: BPtr,
    n: Int,
    kind: Int,
    fill: Float64,
    missing: Float64,
    valid_min: Float64,
    valid_max: Float64,
    flags: Int,
):
    if n < PARALLEL_THRESHOLD:
        mask_range(
            src, mask, 0, n, kind, fill, missing,
            valid_min, valid_max, flags,
        )
        return

    var tasks = (n + PARALLEL_GRAIN - 1) // PARALLEL_GRAIN

    @parameter
    @__copy_capture(
        src, mask, n, kind, fill, missing, valid_min, valid_max, flags
    )
    def work(task: Int):
        var start = task * PARALLEL_GRAIN
        var end = min(start + PARALLEL_GRAIN, n)
        mask_range(
            src, mask, start, end, kind, fill, missing,
            valid_min, valid_max, flags,
        )

    try:
        var ctx = DeviceContext(api="cpu")
        parallelize[work](tasks, ctx=ctx)
    except:
        mask_range(
            src, mask, 0, n, kind, fill, missing,
            valid_min, valid_max, flags,
        )


@export("mnc_unpack_f64")
def mnc_unpack_f64(
    src_addr: Int,
    dst_addr: Int,
    raw_addr: Int,
    mask_addr: Int,
    n: Int,
    kind: Int,
    scale: Float64,
    offset: Float64,
    fill: Float64,
    missing: Float64,
    valid_min: Float64,
    valid_max: Float64,
    flags: Int,
) abi("C") -> Int:
    if n <= 0 or src_addr == 0 or dst_addr == 0:
        return -1
    if kind < 1 or kind > 10:
        return -1
    if flags != 0 and (raw_addr == 0 or mask_addr == 0):
        return -1
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = DPtr(unsafe_from_address=dst_addr)
    decode_values(src, dst, n, kind, scale, offset)
    if flags != 0 and raw_addr != 0 and mask_addr != 0:
        mask_values(
            BPtr(unsafe_from_address=raw_addr),
            BPtr(unsafe_from_address=mask_addr),
            n,
            kind,
            fill,
            missing,
            valid_min,
            valid_max,
            flags,
        )
    return 0


@export("mnc_mask_f64")
def mnc_mask_f64(
    raw_addr: Int,
    mask_addr: Int,
    n: Int,
    kind: Int,
    fill: Float64,
    missing: Float64,
    valid_min: Float64,
    valid_max: Float64,
    flags: Int,
) abi("C") -> Int:
    if n <= 0 or raw_addr == 0 or mask_addr == 0:
        return -1
    if kind < 1 or kind > 10:
        return -1
    mask_values(
        BPtr(unsafe_from_address=raw_addr),
        BPtr(unsafe_from_address=mask_addr),
        n,
        kind,
        fill,
        missing,
        valid_min,
        valid_max,
        flags,
    )
    return 0


@export("mnc_pack_f64")
def mnc_pack_f64(
    src_addr: Int,
    mask_addr: Int,
    dst_addr: Int,
    n: Int,
    kind: Int,
    scale: Float64,
    offset: Float64,
    fill: Float64,
) abi("C") -> Int:
    if n <= 0 or src_addr == 0 or dst_addr == 0 or scale == 0.0:
        return -1
    if kind < 1 or kind > 10:
        return -1
    var src = DPtr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    if mask_addr == 0:
        pack_values(src, dst, n, kind, scale, offset)
        return 0
    var mask = BPtr(unsafe_from_address=mask_addr)
    for i in range(n):
        var value = (src[i] - offset) / scale
        if mask[i] != 0:
            value = fill
        if kind == 1:
            dst.bitcast[Int8]()[i] = Int8(round(value))
        elif kind == 2:
            dst.bitcast[UInt8]()[i] = UInt8(round(value))
        elif kind == 3:
            dst.bitcast[Int16]()[i] = Int16(round(value))
        elif kind == 4:
            dst.bitcast[UInt16]()[i] = UInt16(round(value))
        elif kind == 5:
            dst.bitcast[Int32]()[i] = Int32(round(value))
        elif kind == 6:
            dst.bitcast[UInt32]()[i] = UInt32(round(value))
        elif kind == 7:
            dst.bitcast[Int64]()[i] = Int64(round(value))
        elif kind == 8:
            dst.bitcast[UInt64]()[i] = UInt64(round(value))
        elif kind == 9:
            dst.bitcast[Float32]()[i] = Float32(value)
        else:
            dst.bitcast[Float64]()[i] = value
    return 0


def pack_range(
    src: DPtr,
    dst: BPtr,
    start: Int,
    end: Int,
    kind: Int,
    scale: Float64,
    offset: Float64,
):
    var i = start
    if kind == 1:
        var values = dst.bitcast[Int8]()
        while i + W <= end:
            values.store(
                i,
                round((src.load[width=W](i) - offset) / scale)
                .cast[DType.int8](),
            )
            i += W
        while i < end:
            values[i] = Int8(round((src[i] - offset) / scale))
            i += 1
    elif kind == 2:
        var values = dst.bitcast[UInt8]()
        while i + W <= end:
            values.store(
                i,
                round((src.load[width=W](i) - offset) / scale)
                .cast[DType.uint8](),
            )
            i += W
        while i < end:
            values[i] = UInt8(round((src[i] - offset) / scale))
            i += 1
    elif kind == 3:
        var values = dst.bitcast[Int16]()
        while i + W <= end:
            values.store(
                i,
                round((src.load[width=W](i) - offset) / scale)
                .cast[DType.int16](),
            )
            i += W
        while i < end:
            values[i] = Int16(round((src[i] - offset) / scale))
            i += 1
    elif kind == 4:
        var values = dst.bitcast[UInt16]()
        while i + W <= end:
            values.store(
                i,
                round((src.load[width=W](i) - offset) / scale)
                .cast[DType.uint16](),
            )
            i += W
        while i < end:
            values[i] = UInt16(round((src[i] - offset) / scale))
            i += 1
    elif kind == 5:
        var values = dst.bitcast[Int32]()
        while i + W <= end:
            values.store(
                i,
                round((src.load[width=W](i) - offset) / scale)
                .cast[DType.int32](),
            )
            i += W
        while i < end:
            values[i] = Int32(round((src[i] - offset) / scale))
            i += 1
    elif kind == 6:
        var values = dst.bitcast[UInt32]()
        while i + W <= end:
            values.store(
                i,
                round((src.load[width=W](i) - offset) / scale)
                .cast[DType.uint32](),
            )
            i += W
        while i < end:
            values[i] = UInt32(round((src[i] - offset) / scale))
            i += 1
    elif kind == 7:
        var values = dst.bitcast[Int64]()
        while i + W <= end:
            values.store(
                i,
                round((src.load[width=W](i) - offset) / scale)
                .cast[DType.int64](),
            )
            i += W
        while i < end:
            values[i] = Int64(round((src[i] - offset) / scale))
            i += 1
    elif kind == 8:
        var values = dst.bitcast[UInt64]()
        while i + W <= end:
            values.store(
                i,
                round((src.load[width=W](i) - offset) / scale)
                .cast[DType.uint64](),
            )
            i += W
        while i < end:
            values[i] = UInt64(round((src[i] - offset) / scale))
            i += 1
    elif kind == 9:
        var values = dst.bitcast[Float32]()
        while i + W <= end:
            values.store(
                i,
                ((src.load[width=W](i) - offset) / scale)
                .cast[DType.float32](),
            )
            i += W
        while i < end:
            values[i] = Float32((src[i] - offset) / scale)
            i += 1
    else:
        var values = dst.bitcast[Float64]()
        while i + W <= end:
            values.store(
                i,
                (src.load[width=W](i) - offset) / scale,
            )
            i += W
        while i < end:
            values[i] = (src[i] - offset) / scale
            i += 1


def pack_values(
    src: DPtr,
    dst: BPtr,
    n: Int,
    kind: Int,
    scale: Float64,
    offset: Float64,
):
    pack_range(src, dst, 0, n, kind, scale, offset)


def quantize_range(
    src: DPtr,
    dst: DPtr,
    start: Int,
    end: Int,
    multiplier: Float64,
    inverse: Float64,
):
    var i = start
    while i + W <= end:
        dst.store(
            i,
            round(src.load[width=W](i) * multiplier) * inverse,
        )
        i += W
    while i < end:
        dst[i] = round(src[i] * multiplier) * inverse
        i += 1


@export("mnc_quantize_f64")
def mnc_quantize_f64(
    src_addr: Int, dst_addr: Int, n: Int, multiplier: Float64
) abi("C") -> Int:
    if n <= 0 or src_addr == 0 or dst_addr == 0 or multiplier <= 0.0:
        return -1
    var src = DPtr(unsafe_from_address=src_addr)
    var dst = DPtr(unsafe_from_address=dst_addr)
    var inverse = 1.0 / multiplier
    if n < PARALLEL_THRESHOLD:
        quantize_range(src, dst, 0, n, multiplier, inverse)
        return 0

    var tasks = (n + PARALLEL_GRAIN - 1) // PARALLEL_GRAIN

    @parameter
    @__copy_capture(src, dst, n, multiplier, inverse)
    def work(task: Int):
        var start = task * PARALLEL_GRAIN
        var end = min(start + PARALLEL_GRAIN, n)
        quantize_range(src, dst, start, end, multiplier, inverse)

    try:
        var ctx = DeviceContext(api="cpu")
        parallelize[work](tasks, ctx=ctx)
    except:
        quantize_range(src, dst, 0, n, multiplier, inverse)
    return 0
