"""Benchmark Mojo primitives against Featuretools on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import featuretools.primitives as ft

import mojo_featuretools as mft


def timeit(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def machine():
    model = "unknown CPU"
    try:
        with open("/proc/cpuinfo", encoding="utf8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{model}; {platform.system()} {platform.release()}; Python {platform.python_version()}"


def cases():
    rng = np.random.default_rng(123)

    values_sum = rng.normal(size=10_000_000)
    yield "Sum (10M)", lambda: mft.Sum()(values_sum), lambda: ft.Sum()(values_sum)

    values_cum = rng.normal(size=5_000_000)
    yield "CumSum (5M)", lambda: mft.CumSum()(values_cum), lambda: ft.CumSum()(values_cum)

    values_roll = rng.normal(size=20_000)
    times = pd.date_range("2020-01-01", periods=values_roll.size, freq="s")
    ours_mean = mft.RollingMean(window_length=128, gap=1)
    theirs_mean = ft.RollingMean(window_length=128, gap=1)
    yield (
        "RollingMean (20k, window 128)",
        lambda: ours_mean(times, values_roll),
        lambda: theirs_mean(times, values_roll),
    )

    ours_std = mft.RollingSTD(window_length=128, gap=1)
    theirs_std = ft.RollingSTD(window_length=128, gap=1)
    yield (
        "RollingSTD (20k, window 128)",
        lambda: ours_std(times, values_roll),
        lambda: theirs_std(times, values_roll),
    )

    values_sine = rng.normal(size=5_000_000)
    yield "Sine (5M)", lambda: mft.Sine()(values_sine), lambda: ft.Sine()(values_sine)

    values_group = rng.normal(size=5_000_000)
    codes = rng.integers(0, 1000, size=values_group.size, dtype=np.int64)
    frame = pd.DataFrame({"value": values_group, "code": codes})

    def pandas_groupby():
        return frame.groupby("code").value.agg(
            sum="sum",
            mean="mean",
            min="min",
            max="max",
            std=lambda x: x.std(ddof=0),
            count="count",
        )

    yield (
        "Grouped 6-aggregate (5M, 1k groups)",
        lambda: mft.groupby_aggregate(values_group, codes, 1000),
        pandas_groupby,
    )


def main():
    print(f"Machine: {machine()}")
    print()
    print("| case | Mojo | Featuretools/pandas | speedup | result |")
    print("| --- | ---: | ---: | ---: | --- |")
    for name, mojo_function, reference_function in cases():
        mojo_function()
        reference_function()
        mojo_time = timeit(mojo_function)
        reference_time = timeit(reference_function)
        speedup = reference_time / mojo_time
        result = "faster" if speedup >= 1 else "slower"
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{reference_time * 1e3:.2f} ms | {speedup:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()
