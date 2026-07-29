"""Numeric Featuretools primitives over caller-owned float64 buffers."""

from std.algorithm import parallelize
from std.gpu.host import DeviceContext
from std.math import abs, cos, isnan, log, sin, sqrt, tan
from std.sys.info import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime PARALLEL_UNARY_THRESHOLD = 262_144
comptime PARALLEL_UNARY_CHUNKS = 16


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def nan_value() -> Float64:
    var zero = 0.0
    return zero / zero


def sine_range(src: FPtr, dst: FPtr, start: Int, end: Int):
    comptime W = simdwidthof[DType.float64]()
    var stop = start + (end - start) // W * W
    for i in range(start, stop, W):
        var values = src.load[width=W](i)
        dst.store(i, sin(values))
    for i in range(stop, end):
        dst[i] = sin(src[i])


def sine_transform(src: FPtr, dst: FPtr, n: Int):
    if n < PARALLEL_UNARY_THRESHOLD:
        sine_range(src, dst, 0, n)
        return

    var chunk_size = (n + PARALLEL_UNARY_CHUNKS - 1) // PARALLEL_UNARY_CHUNKS

    @parameter
    def work(chunk: Int):
        var start = chunk * chunk_size
        var end = min(start + chunk_size, n)
        if start < end:
            sine_range(src, dst, start, end)

    try:
        var ctx = DeviceContext(api="cpu")
        parallelize[work](PARALLEL_UNARY_CHUNKS, ctx)
    except:
        sine_range(src, dst, 0, n)


def unary(src: FPtr, dst: FPtr, n: Int, op: Int):
    if op == 3:
        sine_transform(src, dst, n)
        return
    for i in range(n):
        var value = src[i]
        if op == 0:
            dst[i] = abs(value)
        elif op == 1:
            dst[i] = sqrt(value)
        elif op == 2:
            dst[i] = log(value)
        elif op == 4:
            dst[i] = cos(value)
        else:
            dst[i] = tan(value)


def reduce_sum(src: FPtr, n: Int) -> Float64:
    comptime W = simdwidthof[DType.float64]()
    var vector_total = SIMD[DType.float64, W](0.0)
    var stop = n - n % W
    for i in range(0, stop, W):
        var values = src.load[width=W](i)
        vector_total += isnan(values).select(0.0, values)
    var total = vector_total.reduce_add()
    for i in range(stop, n):
        var value = src[i]
        if not isnan(value):
            total += value
    return total


def reduce(src: FPtr, n: Int, op: Int, skipna: Bool) -> Float64:
    if op == 0 and skipna:
        return reduce_sum(src, n)
    var total = 0.0
    var count = 0
    var extreme = 0.0
    for i in range(n):
        var value = src[i]
        if isnan(value):
            if not skipna:
                return nan_value()
            continue
        if count == 0:
            extreme = value
        elif op == 2 and value < extreme:
            extreme = value
        elif op == 3 and value > extreme:
            extreme = value
        total += value
        count += 1
    if op == 0:
        return total
    if op == 7:
        return Float64(count)
    if op == 8:
        return total / Float64(n) if n > 0 else nan_value()
    if count == 0:
        return nan_value()
    if op == 1:
        return total / Float64(count)
    if op == 2 or op == 3:
        return extreme

    var mean = total / Float64(count)
    var m2 = 0.0
    var m3 = 0.0
    for i in range(n):
        var value = src[i]
        if not isnan(value):
            var delta = value - mean
            m2 += delta * delta
            if op == 6:
                m3 += delta * delta * delta
    if op == 4:
        return sqrt(m2 / Float64(count))
    if op == 5:
        return m2 / Float64(count)
    if count < 3 or m2 == 0.0:
        return 0.0 if m2 == 0.0 and count >= 3 else nan_value()
    var sample_std = sqrt(m2 / Float64(count - 1))
    return (
        Float64(count)
        * m3
        / (
            Float64((count - 1) * (count - 2))
            * sample_std
            * sample_std
            * sample_std
        )
    )


def select_value(src: FPtr, n: Int, kth: Int) -> Float64:
    var left = 0
    var right = n - 1
    while left < right:
        var pivot = src[(left + right) // 2]
        var i = left
        var j = right
        while i <= j:
            while src[i] < pivot:
                i += 1
            while src[j] > pivot:
                j -= 1
            if i <= j:
                var swap = src[i]
                src[i] = src[j]
                src[j] = swap
                i += 1
                j -= 1
        if kth <= j:
            right = j
        elif kth >= i:
            left = i
        else:
            break
    return src[kth]


def median(src: FPtr, n: Int) -> Float64:
    var count = 0
    for i in range(n):
        var value = src[i]
        if not isnan(value):
            src[count] = value
            count += 1
    if count == 0:
        return nan_value()
    var high = select_value(src, count, count // 2)
    if count % 2 == 0:
        var low = select_value(src, count, count // 2 - 1)
        return 0.5 * low + 0.5 * high
    return high


def cumulative(src: FPtr, dst: FPtr, n: Int, op: Int):
    var state = 0.0
    var have_value = False
    for i in range(n):
        var value = src[i]
        if isnan(value):
            dst[i] = nan_value()
            continue
        if not have_value:
            state = value
            have_value = True
        elif op == 0 or op == 1:
            state += value
        elif op == 2 and value < state:
            state = value
        elif op == 3 and value > state:
            state = value
        dst[i] = state / Float64(i + 1) if op == 1 else state


def diff(src: FPtr, dst: FPtr, n: Int, periods: Int):
    for i in range(n):
        var current = i - periods
        var previous = current - 1
        if (
            i == 0
            or current < 0
            or current >= n
            or previous < 0
            or previous >= n
        ):
            dst[i] = nan_value()
        else:
            dst[i] = src[current] - src[previous]


def rolling_sum_mean(
    src: FPtr,
    dst: FPtr,
    n: Int,
    window: Int,
    gap: Int,
    min_periods: Int,
):
    var total = 0.0
    var count = 0
    for i in range(n):
        var entering = i - gap
        if entering >= 0:
            var value = src[entering]
            if not isnan(value):
                total += value
                count += 1
        var leaving = entering - window
        if leaving >= 0:
            var value = src[leaving]
            if not isnan(value):
                total -= value
                count -= 1
        if count > 0 and count >= min_periods:
            dst[i] = total / Float64(count)


def rolling_std(
    src: FPtr,
    dst: FPtr,
    n: Int,
    window: Int,
    gap: Int,
    min_periods: Int,
):
    var anchor = 0.0
    for i in range(n):
        if not isnan(src[i]):
            anchor = src[i]
            break
    var total = 0.0
    var total2 = 0.0
    var count = 0
    for i in range(n):
        var entering = i - gap
        var leaving = entering - window
        if leaving >= 0:
            var value = src[leaving]
            if not isnan(value):
                var centered = value - anchor
                total -= centered
                total2 -= centered * centered
                count -= 1
        if entering >= 0:
            var value = src[entering]
            if not isnan(value):
                var centered = value - anchor
                total += centered
                total2 += centered * centered
                count += 1
        if count > 1 and count >= min_periods:
            var variance = (total2 - total * total / Float64(count)) / Float64(
                count - 1
            )
            if variance < 0.0:
                variance = 0.0
            dst[i] = sqrt(variance)


def rolling_extreme(
    src: FPtr,
    dst: FPtr,
    scratch: IPtr,
    n: Int,
    window: Int,
    gap: Int,
    min_periods: Int,
    find_max: Bool,
):
    var head = 0
    var tail = 0
    var count = 0
    for i in range(n):
        var entering = i - gap
        if entering >= 0:
            var value = src[entering]
            if not isnan(value):
                count += 1
                while head < tail:
                    var last = Int(scratch[(tail - 1) % n])
                    if (find_max and src[last] > value) or (
                        not find_max and src[last] < value
                    ):
                        break
                    tail -= 1
                scratch[tail % n] = Int64(entering)
                tail += 1
        var leaving = entering - window
        if leaving >= 0 and not isnan(src[leaving]):
            count -= 1
        while head < tail and Int(scratch[head % n]) <= leaving:
            head += 1
        if count > 0 and count >= min_periods and head < tail:
            dst[i] = src[Int(scratch[head % n])]


def rolling(
    src: FPtr,
    dst: FPtr,
    scratch: IPtr,
    n: Int,
    window: Int,
    gap: Int,
    min_periods: Int,
    op: Int,
):
    for i in range(n):
        dst[i] = nan_value()
    if op == 0:
        rolling_sum_mean(src, dst, n, window, gap, min_periods)
    elif op == 1:
        rolling_std(src, dst, n, window, gap, min_periods)
    else:
        rolling_extreme(src, dst, scratch, n, window, gap, min_periods, op == 3)


def expanding(
    src: FPtr,
    dst: FPtr,
    n: Int,
    gap: Int,
    min_periods: Int,
    op: Int,
):
    var total = 0.0
    var mean = 0.0
    var m2 = 0.0
    var extreme = 0.0
    var count = 0
    for i in range(n):
        dst[i] = nan_value()
        var entering = i - gap
        if entering < 0:
            continue
        var value = src[entering]
        if not isnan(value):
            count += 1
            total += value
            if count == 1:
                mean = value
                extreme = value
            else:
                var delta = value - mean
                mean += delta / Float64(count)
                m2 += delta * (value - mean)
                if op == 2 and value < extreme:
                    extreme = value
                elif op == 3 and value > extreme:
                    extreme = value
        if count >= min_periods and count > 0:
            if op == 0:
                dst[i] = total / Float64(count)
            elif op == 1 and count > 1:
                dst[i] = sqrt(m2 / Float64(count - 1))
            elif op == 2 or op == 3:
                dst[i] = extreme


def group_reduce(
    values: FPtr,
    codes: IPtr,
    sums: FPtr,
    means: FPtr,
    mins: FPtr,
    maxs: FPtr,
    stds: FPtr,
    counts: IPtr,
    n: Int,
    groups: Int,
):
    for group in range(groups):
        sums[group] = 0.0
        means[group] = 0.0
        mins[group] = nan_value()
        maxs[group] = nan_value()
        stds[group] = nan_value()
        counts[group] = 0
    for i in range(n):
        var group = Int(codes[i])
        var value = values[i]
        if group < 0 or group >= groups or isnan(value):
            continue
        var count = Int(counts[group]) + 1
        counts[group] = Int64(count)
        sums[group] += value
        if count == 1:
            means[group] = value
            mins[group] = value
            maxs[group] = value
            stds[group] = 0.0
        else:
            var old_mean = means[group]
            var delta = value - old_mean
            means[group] += delta / Float64(count)
            stds[group] += delta * (value - means[group])
            if value < mins[group]:
                mins[group] = value
            if value > maxs[group]:
                maxs[group] = value
    for group in range(groups):
        var count = Int(counts[group])
        if count == 0:
            means[group] = nan_value()
        if count > 0:
            stds[group] = sqrt(stds[group] / Float64(count))
        else:
            stds[group] = nan_value()


@export("mft_unary")
def mft_unary(src: Int, dst: Int, n: Int, op: Int) abi("C"):
    unary(fp(src), fp(dst), n, op)


@export("mft_reduce")
def mft_reduce(src: Int, n: Int, op: Int, skipna: Int) abi("C") -> Float64:
    return reduce(fp(src), n, op, skipna != 0)


@export("mft_median")
def mft_median(src: Int, n: Int) abi("C") -> Float64:
    return median(fp(src), n)


@export("mft_cumulative")
def mft_cumulative(src: Int, dst: Int, n: Int, op: Int) abi("C"):
    cumulative(fp(src), fp(dst), n, op)


@export("mft_diff")
def mft_diff(src: Int, dst: Int, n: Int, periods: Int) abi("C"):
    diff(fp(src), fp(dst), n, periods)


@export("mft_rolling")
def mft_rolling(
    src: Int,
    dst: Int,
    scratch: Int,
    n: Int,
    window: Int,
    gap: Int,
    min_periods: Int,
    op: Int,
) abi("C"):
    rolling(fp(src), fp(dst), ip(scratch), n, window, gap, min_periods, op)


@export("mft_expanding")
def mft_expanding(
    src: Int,
    dst: Int,
    n: Int,
    gap: Int,
    min_periods: Int,
    op: Int,
) abi("C"):
    expanding(fp(src), fp(dst), n, gap, min_periods, op)


@export("mft_group_reduce")
def mft_group_reduce(
    values: Int,
    codes: Int,
    sums: Int,
    means: Int,
    mins: Int,
    maxs: Int,
    stds: Int,
    counts: Int,
    n: Int,
    groups: Int,
) abi("C"):
    group_reduce(
        fp(values),
        ip(codes),
        fp(sums),
        fp(means),
        fp(mins),
        fp(maxs),
        fp(stds),
        ip(counts),
        n,
        groups,
    )
