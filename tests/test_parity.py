"""Behavioral and numerical parity with Featuretools 1.31 primitives."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import featuretools.primitives as upstream

import mojo_featuretools as mft
from mojo_featuretools._lib import addr, f64, i64


@pytest.fixture(scope="module")
def numeric():
    rng = np.random.default_rng(42)
    values = rng.normal(size=257)
    values[[0, 9, 80, 256]] = np.nan
    return values


@pytest.mark.parametrize(
    "name",
    ["Sum", "Min", "Max", "Std", "Variance", "Skew", "Median", "Count"],
)
def test_aggregation_parity(name, numeric):
    actual = getattr(mft, name)()(numeric)
    expected = getattr(upstream, name)()(numeric)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12, nan_ok=True)


@pytest.mark.parametrize("skipna", [True, False])
def test_mean_parity(skipna, numeric):
    actual = mft.Mean(skipna=skipna)(numeric)
    expected = upstream.Mean(skipna=skipna)(numeric)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12, nan_ok=True)


@pytest.mark.parametrize(
    "values",
    [
        [True, False, True, None, True],
        [False, False, np.nan],
        [],
    ],
)
def test_percent_true_parity(values):
    actual = mft.PercentTrue()(values)
    expected = upstream.PercentTrue()(values)
    assert actual == pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize("name", ["Sum", "Mean", "Min", "Max", "Std", "Variance", "Median"])
def test_empty_and_all_null_aggregation_parity(name):
    for values in ([], [None, np.nan]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actual = getattr(mft, name)()(values)
            expected = getattr(upstream, name)()(values)
        assert actual == pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize(
    "name",
    ["Absolute", "SquareRoot", "NaturalLogarithm", "Sine", "Cosine", "Tangent"],
)
def test_unary_transform_parity(name):
    values = np.array([np.nan, 0.125, 0.5, 1.0, 2.0, 8.0])
    if name == "Absolute":
        values[1] = -2.0
    actual = getattr(mft, name)()(values)
    expected = getattr(upstream, name)()(values)
    assert isinstance(actual, pd.Series)
    assert np.allclose(actual, expected, equal_nan=True, rtol=2e-10, atol=2e-12)


def test_sum_simd_tail_skips_nulls():
    values = np.linspace(-4.0, 7.0, 259)
    values[[3, 256, 258]] = np.nan
    actual = mft.Sum()(values)
    expected = upstream.Sum()(values)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_sine_simd_tail_and_parallel_threshold():
    for size in (259, 262_143, 262_147):
        values = np.linspace(-8.0, 8.0, size)
        values[[0, size - 2]] = np.nan
        actual = mft.Sine()(values)
        expected = np.sin(values)
        assert np.allclose(
            actual, expected, equal_nan=True, rtol=2e-10, atol=2e-12
        )


def test_numpy_float64_input_stays_zero_copy():
    values = np.arange(31, dtype=np.float64)
    native = f64(values)
    assert native is values
    assert np.shares_memory(native, values)


def test_ffi_converters_enforce_shape_dtype_and_integer_precision():
    with pytest.raises(ValueError, match="one-dimensional"):
        f64(np.ones((2, 2)))
    with pytest.raises(ValueError, match="represented exactly"):
        f64(np.array([2**53 + 1], dtype=np.int64))
    with pytest.raises(ValueError, match="wider than float64"):
        f64(np.array([1], dtype=np.longdouble))
    with pytest.raises(TypeError, match="integer dtype"):
        i64([0.0, 1.5])
    with pytest.raises(TypeError, match="C-contiguous"):
        addr(np.arange(8, dtype=np.float64)[::2])


def test_noncontiguous_input_is_copied_safely():
    values = np.arange(20, dtype=np.float64)[::2]
    assert not values.flags.c_contiguous
    assert mft.Sum()(values) == pytest.approx(values.sum())


@pytest.mark.parametrize("name", ["CumSum", "CumMean", "CumMin", "CumMax"])
def test_cumulative_parity(name, numeric):
    actual = getattr(mft, name)()(numeric)
    expected = getattr(upstream, name)()(numeric)
    assert np.allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("periods", [-2, -1, 0, 1, 3])
def test_diff_parity(periods, numeric):
    actual = mft.Diff(periods=periods)(numeric)
    expected = upstream.Diff(periods=periods)(numeric)
    assert np.allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize(
    "name", ["RollingMean", "RollingSTD", "RollingMin", "RollingMax"]
)
@pytest.mark.parametrize(
    "window,gap,min_periods", [(3, 1, 1), (11, 0, 5), (32, 4, 0)]
)
def test_rolling_parity(name, window, gap, min_periods, numeric):
    times = pd.date_range("2024-01-01", periods=numeric.size, freq="min")
    ours = getattr(mft, name)(
        window_length=window, gap=gap, min_periods=min_periods
    )
    theirs = getattr(upstream, name)(
        window_length=window, gap=gap, min_periods=min_periods
    )
    actual = ours(times, numeric)
    expected = theirs(times, numeric)
    assert np.allclose(actual, expected, equal_nan=True, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize(
    "name", ["ExpandingMean", "ExpandingSTD", "ExpandingMin", "ExpandingMax"]
)
@pytest.mark.parametrize("gap,min_periods", [(0, 1), (1, 1), (4, 8)])
def test_expanding_parity(name, gap, min_periods, numeric):
    times = pd.date_range("2024-01-01", periods=numeric.size, freq="min")
    actual = getattr(mft, name)(gap=gap, min_periods=min_periods)(times, numeric)
    expected = getattr(upstream, name)(gap=gap, min_periods=min_periods)(
        times, numeric
    )
    assert np.allclose(actual, expected, equal_nan=True, rtol=2e-11, atol=2e-12)


def test_rolling_std_is_stable_for_large_offsets():
    rng = np.random.default_rng(9)
    values = 1e9 + rng.normal(size=500)
    times = pd.date_range("2024-01-01", periods=values.size, freq="s")
    actual = mft.RollingSTD(64, gap=0)(times, values)
    expected = upstream.RollingSTD(64, gap=0)(times, values)
    assert np.allclose(actual, expected, equal_nan=True, rtol=2e-7, atol=2e-7)


def test_groupby_aggregate_matches_pandas():
    rng = np.random.default_rng(5)
    values = rng.normal(size=2000)
    values[::37] = np.nan
    codes = rng.integers(-1, 17, size=values.size, dtype=np.int64)
    actual = mft.groupby_aggregate(values, codes, n_groups=17)

    frame = pd.DataFrame({"value": values, "code": codes})
    expected = frame[frame.code >= 0].groupby("code").value.agg(
        sum="sum",
        mean="mean",
        min="min",
        max="max",
        std=lambda x: x.std(ddof=0),
        count="count",
    )
    expected = expected.reindex(range(17))
    assert np.allclose(
        actual[["sum", "mean", "min", "max", "std"]],
        expected[["sum", "mean", "min", "max", "std"]],
        equal_nan=True,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.array_equal(actual["count"], expected["count"])


def test_groupby_empty_groups_and_inferred_count():
    actual = mft.groupby_aggregate([1.0, np.nan, 4.0], [0, 2, 2])
    assert len(actual) == 3
    assert actual.loc[1, "sum"] == 0.0
    assert np.isnan(actual.loc[1, "mean"])
    assert actual.loc[1, "count"] == 0


def test_groupby_rejects_lossy_or_out_of_range_codes():
    with pytest.raises(TypeError, match="integer dtype"):
        mft.groupby_aggregate([1.0, 2.0], [0.0, 1.5])
    with pytest.raises(ValueError, match="smaller than n_groups"):
        mft.groupby_aggregate([1.0, 2.0], [0, 2], n_groups=2)
    with pytest.raises(TypeError, match="n_groups must be an integer"):
        mft.groupby_aggregate([1.0], [0], n_groups=1.5)


def test_primitive_metadata_and_function_contract():
    primitive = mft.RollingMean(window_length=5, gap=0, min_periods=2)
    assert primitive.name == "rolling_mean"
    assert primitive.uses_full_dataframe
    assert primitive.get_arguments() == [
        ("window_length", 5),
        ("gap", 0),
        ("min_periods", 2),
    ]
    assert callable(primitive.get_function())


def test_time_offset_windows_fail_clearly():
    with pytest.raises(NotImplementedError, match="integer row windows"):
        mft.RollingMean(window_length="3D", gap="1D")


def test_integer_parameters_are_not_silently_narrowed():
    with pytest.raises(TypeError, match="periods must be an integer"):
        mft.Diff(periods=1.5)
    with pytest.raises(TypeError, match="gap and min_periods must be integers"):
        mft.ExpandingMean(gap=1.5)
