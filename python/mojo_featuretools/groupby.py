"""Batched numeric grouped aggregation for DFS-style workloads."""

from __future__ import annotations

from operator import index

import numpy as np
import pandas as pd

from ._lib import addr, f64, i64, lib


def groupby_aggregate(values, codes, n_groups=None):
    """Compute sum, mean, min, max, std, and count for integer group codes.

    Negative codes are treated as missing groups. Standard deviation uses the
    population convention of Featuretools' ``Std`` primitive.
    """

    values_array = f64(values)
    codes_array = i64(codes)
    if values_array.ndim != 1 or codes_array.ndim != 1:
        raise ValueError("values and codes must be one-dimensional")
    if values_array.size != codes_array.size:
        raise ValueError("values and codes must have equal length")
    valid = codes_array[codes_array >= 0]
    if n_groups is None:
        groups = int(valid.max()) + 1 if valid.size else 0
    else:
        try:
            groups = index(n_groups)
        except TypeError:
            raise TypeError("n_groups must be an integer") from None
    if groups < 0:
        raise ValueError("n_groups must be nonnegative")
    if valid.size and int(valid.max()) >= groups:
        raise ValueError("group code must be smaller than n_groups")

    sums = np.empty(groups, dtype=np.float64)
    means = np.empty(groups, dtype=np.float64)
    mins = np.empty(groups, dtype=np.float64)
    maxs = np.empty(groups, dtype=np.float64)
    stds = np.empty(groups, dtype=np.float64)
    counts = np.empty(groups, dtype=np.int64)
    if groups:
        lib().mft_group_reduce(
            addr(values_array),
            addr(codes_array),
            addr(sums),
            addr(means),
            addr(mins),
            addr(maxs),
            addr(stds),
            addr(counts),
            values_array.size,
            groups,
        )
    return pd.DataFrame(
        {
            "sum": sums,
            "mean": means,
            "min": mins,
            "max": maxs,
            "std": stds,
            "count": counts,
        }
    )
