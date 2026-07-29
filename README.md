# mojo-featuretools

`mojo-featuretools` is a standalone Mojo port of the compute-heavy numeric
primitives used by [Featuretools](https://github.com/alteryx/featuretools).
It keeps the covered class names, constructor signatures, callable behavior,
null handling, and population/sample variance conventions of Featuretools
1.31.0 while moving the array loops into one compiled Mojo shared library.

This is a primitive-layer port, not a reimplementation of the EntitySet,
feature graph, or Deep Feature Synthesis planner. It is useful when primitives
are called directly and when numeric values have already been encoded into
contiguous arrays.

## Coverage

The public package is `mojo_featuretools`. The following 29 primitive classes
are covered:

- Aggregation: `Sum`, `Mean`, `Min`, `Max`, `Std`, `Variance`, `Skew`,
  `Median`, `Count`, and `PercentTrue`.
- Numeric transform: `Absolute`, `SquareRoot`, `NaturalLogarithm`, `Sine`,
  `Cosine`, and `Tangent`.
- Full-series transform: `CumSum`, `CumMean`, `CumMin`, `CumMax`, and `Diff`.
- Time series: `RollingMean`, `RollingSTD`, `RollingMin`, `RollingMax`,
  `ExpandingMean`, `ExpandingSTD`, `ExpandingMin`, and `ExpandingMax`.

`groupby_aggregate` is an additional batched DFS-oriented operation. In one
pass it computes sum, mean, minimum, maximum, population standard deviation,
and count for integer group codes.

Not covered:

- EntitySet construction, relationships, feature graph generation, DFS search,
  cutoff-time orchestration, and distributed backends.
- Categorical, text, URL, geospatial, and datetime-extraction primitives.
- Rolling windows expressed as pandas offset strings. The rolling classes
  currently accept integer row windows and integer gaps only.
- Featuretools' Woodwork schemas and its plugin discovery machinery.

The covered direct primitive calls are API-compatible; the classes are not
subclasses of Featuretools' own primitive base classes and therefore cannot be
inserted into the upstream DFS planner as custom primitives.

## Install

The repository pins the tested Mojo nightly. Install the environment and build
the shared library:

```bash
pixi install
pixi run build
```

Run the parity suite and benchmarks with:

```bash
pixi run test
pixi run bench
```

## Usage

```python
import numpy as np
import pandas as pd
import mojo_featuretools as ft

values = np.array([4.0, 3.0, np.nan, 2.0, 1.0])
times = pd.date_range("2024-01-01", periods=len(values), freq="min")

print(ft.Mean()(values))
# 2.5

print(ft.RollingMean(window_length=3, gap=1)(times, values).tolist())
# [nan, 4.0, 3.5, 3.5, 2.5]

groups = np.array([0, 0, 1, 1, 1])
print(ft.groupby_aggregate(values, groups, n_groups=2))
```

Featuretools-style imports are also available:

```python
from mojo_featuretools.primitives import CumSum, RollingSTD, Sum
```

## Benchmarks

Measured with `pixi run bench`; each number is the best of five warmed runs.
The reference column calls the equivalent Featuretools 1.31.0 primitive, or
pandas for the extra grouped operation.

Machine: Intel Xeon E5-2697 v4 at 2.30 GHz, Linux 6.8.0-136-generic,
Python 3.13.14.

| case | Mojo | Featuretools/pandas | speedup | result |
| --- | ---: | ---: | ---: | --- |
| Sum (10M) | 10.10 ms | 94.57 ms | 9.36x | faster |
| CumSum (5M) | 33.45 ms | 117.55 ms | 3.51x | faster |
| RollingMean (20k, window 128) | 0.10 ms | 1699.85 ms | 16753.27x | faster |
| RollingSTD (20k, window 128) | 0.23 ms | 2119.25 ms | 9273.89x | faster |
| Sine (5M) | 20.96 ms | 128.95 ms | 6.15x | faster |
| Grouped 6-aggregate (5M, 1k groups) | 120.33 ms | 808.85 ms | 6.72x | faster |

Sum uses a NaN-masked SIMD reduction. Sine uses SIMD with scalar remainder
handling and switches to thresholded CPU parallelism for large arrays. NumPy
`float64` inputs remain zero-copy through the pandas-compatible wrapper and
the FFI boundary.

No GPU path is included. Sine is the only covered kernel with enough arithmetic
intensity to be a candidate, but the pinned Mojo NVIDIA backend rejects
`Float64` sine. Casting through `Float32` would violate the existing parity
tolerance. The remaining kernels are memory-bound or sequential prefix/window
operations, so their transfer overhead does not justify GPU execution.

## How it works

All kernels live in one Mojo compilation unit and build to
`dist/libmojo-featuretools.so`. Python owns every input, output, and scratch
allocation. Contiguous NumPy `float64` and `int64` buffers cross the C ABI as
integer addresses; Mojo reconstructs mutable `UnsafePointer` values internally.
No Python objects cross the boundary. The CPU parallel runtime manages worker
state, while all numeric buffers remain caller-owned.

The wrapper converts Featuretools-style list or pandas inputs to contiguous
numeric buffers, makes one `ctypes` call, and returns either a scalar, a pandas
Series, or a NumPy array matching the corresponding upstream primitive. Nulls
are represented as IEEE `NaN`. Rolling means use a sliding sum, rolling
extrema use monotonic deques, and rolling standard deviation maintains
centered first and second moments for stable linear-time evaluation.

## License

MIT.
