"""Featuretools-compatible numeric primitive classes backed by Mojo."""

from __future__ import annotations

from inspect import signature
from operator import index

import numpy as np
import pandas as pd

from ._lib import addr, f64, lib


class PrimitiveBase:
    name: str | None = None
    default_value = np.nan
    uses_full_dataframe = False
    number_output_features = 1
    commutative = False

    def __call__(self, *args, **kwargs):
        try:
            method = self._method
        except AttributeError:
            method = self.get_function()
            self._method = method
        return method(*args, **kwargs)

    def get_function(self):
        raise NotImplementedError

    def get_arguments(self):
        values = []
        for name, parameter in signature(self.__class__).parameters.items():
            value = getattr(self, name)
            if type(value) is type(parameter.default) and value == parameter.default:
                continue
            values.append((name, value))
        return values

    def get_args_string(self):
        arguments = [f"{name}={value}" for name, value in self.get_arguments()]
        return f", {', '.join(arguments)}" if arguments else ""

    def generate_name(self, base_feature_names):
        return f"{self.name.upper()}({', '.join(base_feature_names)}{self.get_args_string()})"

    def generate_names(self, base_feature_names):
        return [self.generate_name(base_feature_names)]


class AggregationPrimitive(PrimitiveBase):
    pass


class TransformPrimitive(PrimitiveBase):
    pass


def _as_series(values):
    if isinstance(values, pd.Series):
        return values
    return pd.Series(values, copy=False)


def _reduce_function(op: int, skipna: bool = True):
    def apply(values):
        array = f64(values)
        return lib().mft_reduce(addr(array), array.size, op, int(skipna))

    return apply


class Sum(AggregationPrimitive):
    name = "sum"
    default_value = 0

    def get_function(self):
        return _reduce_function(0)


class Mean(AggregationPrimitive):
    name = "mean"

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        return _reduce_function(1, self.skipna)


class Min(AggregationPrimitive):
    name = "min"

    def get_function(self):
        return _reduce_function(2)


class Max(AggregationPrimitive):
    name = "max"

    def get_function(self):
        return _reduce_function(3)


class Std(AggregationPrimitive):
    name = "std"

    def get_function(self):
        return _reduce_function(4)


class Variance(AggregationPrimitive):
    name = "variance"

    def get_function(self):
        return _reduce_function(5)


class Skew(AggregationPrimitive):
    name = "skew"

    def get_function(self):
        return _reduce_function(6)


class Count(AggregationPrimitive):
    name = "count"
    default_value = 0

    def get_function(self):
        native = _reduce_function(7)
        return lambda values: int(native(values))


class PercentTrue(AggregationPrimitive):
    name = "percent_true"
    default_value = pd.NA

    def get_function(self):
        return _reduce_function(8)


class Median(AggregationPrimitive):
    name = "median"

    def get_function(self):
        def apply(values):
            array = f64(values, copy=True)
            return lib().mft_median(addr(array), array.size)

        return apply


def _unary_function(op: int):
    def apply(values):
        series = _as_series(values)
        array = f64(series)
        result = np.empty(array.size, dtype=np.float64)
        lib().mft_unary(addr(array), addr(result), array.size, op)
        return pd.Series(result, index=series.index, copy=False)

    return apply


class Absolute(TransformPrimitive):
    name = "absolute"

    def get_function(self):
        return _unary_function(0)


class SquareRoot(TransformPrimitive):
    name = "square_root"

    def get_function(self):
        return _unary_function(1)


class NaturalLogarithm(TransformPrimitive):
    name = "natural_logarithm"

    def get_function(self):
        return _unary_function(2)


class Sine(TransformPrimitive):
    name = "sine"

    def get_function(self):
        return _unary_function(3)


class Cosine(TransformPrimitive):
    name = "cosine"

    def get_function(self):
        return _unary_function(4)


class Tangent(TransformPrimitive):
    name = "tangent"

    def get_function(self):
        return _unary_function(5)


def _cumulative_function(op: int):
    def apply(values):
        series = _as_series(values)
        array = f64(series)
        result = np.empty(array.size, dtype=np.float64)
        lib().mft_cumulative(addr(array), addr(result), array.size, op)
        return pd.Series(result, index=series.index, copy=False)

    return apply


class CumSum(TransformPrimitive):
    name = "cum_sum"
    uses_full_dataframe = True

    def get_function(self):
        return _cumulative_function(0)


class CumMean(TransformPrimitive):
    name = "cum_mean"
    uses_full_dataframe = True

    def get_function(self):
        return _cumulative_function(1)


class CumMin(TransformPrimitive):
    name = "cum_min"
    uses_full_dataframe = True

    def get_function(self):
        return _cumulative_function(2)


class CumMax(TransformPrimitive):
    name = "cum_max"
    uses_full_dataframe = True

    def get_function(self):
        return _cumulative_function(3)


class Diff(TransformPrimitive):
    name = "diff"
    uses_full_dataframe = True

    def __init__(self, periods=0):
        try:
            self.periods = index(periods)
        except TypeError:
            raise TypeError("periods must be an integer") from None

    def get_function(self):
        def apply(values):
            series = _as_series(values)
            array = f64(series)
            result = np.empty(array.size, dtype=np.float64)
            lib().mft_diff(addr(array), addr(result), array.size, self.periods)
            return pd.Series(result, index=series.index, copy=False)

        return apply


class _Rolling(TransformPrimitive):
    uses_full_dataframe = True
    _op = 0

    def __init__(self, window_length=3, gap=1, min_periods=1):
        try:
            window_length = index(window_length)
            gap = index(gap)
            min_periods = index(min_periods)
        except TypeError:
            raise NotImplementedError(
                "Mojo rolling primitives currently support integer row windows only"
            ) from None
        if window_length <= 0 or gap < 0 or min_periods < 0:
            raise ValueError("window_length must be positive; gap and min_periods nonnegative")
        if min_periods > window_length:
            raise ValueError("min_periods must be no greater than window_length")
        self.window_length = window_length
        self.gap = gap
        self.min_periods = min_periods

    def get_function(self):
        def apply(datetime, numeric):
            if len(datetime) != len(numeric):
                raise ValueError("datetime and numeric inputs must have equal length")
            array = f64(numeric)
            result = np.empty(array.size, dtype=np.float64)
            scratch = np.empty(max(1, array.size), dtype=np.int64)
            lib().mft_rolling(
                addr(array),
                addr(result),
                addr(scratch),
                array.size,
                self.window_length,
                self.gap,
                self.min_periods,
                self._op,
            )
            return result

        return apply


class RollingMean(_Rolling):
    name = "rolling_mean"
    _op = 0

    def __init__(self, window_length=3, gap=1, min_periods=0):
        super().__init__(window_length, gap, min_periods)


class RollingSTD(_Rolling):
    name = "rolling_std"
    _op = 1


class RollingMin(_Rolling):
    name = "rolling_min"
    _op = 2


class RollingMax(_Rolling):
    name = "rolling_max"
    _op = 3


class _Expanding(TransformPrimitive):
    uses_full_dataframe = True
    _op = 0

    def __init__(self, gap=1, min_periods=1):
        try:
            gap = index(gap)
            min_periods = index(min_periods)
        except TypeError:
            raise TypeError("gap and min_periods must be integers") from None
        if gap < 0 or min_periods < 0:
            raise ValueError("gap and min_periods must be nonnegative")
        self.gap = gap
        self.min_periods = min_periods

    def get_function(self):
        def apply(datetime, numeric):
            if len(datetime) != len(numeric):
                raise ValueError("datetime and numeric inputs must have equal length")
            array = f64(numeric)
            result = np.empty(array.size, dtype=np.float64)
            lib().mft_expanding(
                addr(array),
                addr(result),
                array.size,
                self.gap,
                self.min_periods,
                self._op,
            )
            return result

        return apply


class ExpandingMean(_Expanding):
    name = "expanding_mean"
    _op = 0


class ExpandingSTD(_Expanding):
    name = "expanding_std"
    _op = 1


class ExpandingMin(_Expanding):
    name = "expanding_min"
    _op = 2


class ExpandingMax(_Expanding):
    name = "expanding_max"
    _op = 3


__all__ = [
    "Absolute",
    "AggregationPrimitive",
    "Cosine",
    "Count",
    "CumMax",
    "CumMean",
    "CumMin",
    "CumSum",
    "Diff",
    "ExpandingMax",
    "ExpandingMean",
    "ExpandingMin",
    "ExpandingSTD",
    "Max",
    "Mean",
    "Median",
    "Min",
    "NaturalLogarithm",
    "PercentTrue",
    "PrimitiveBase",
    "RollingMax",
    "RollingMean",
    "RollingMin",
    "RollingSTD",
    "Sine",
    "Skew",
    "SquareRoot",
    "Std",
    "Sum",
    "Tangent",
    "TransformPrimitive",
    "Variance",
]
