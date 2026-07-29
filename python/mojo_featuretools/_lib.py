"""Load the Mojo shared library and declare its C ABI."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_FEATURETOOLS_LIB") or os.path.join(
    ROOT, "dist", "libmojo-featuretools.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mft_unary": ([I, I, I, I], None),
    "mft_reduce": ([I, I, I, I], F),
    "mft_median": ([I, I], F),
    "mft_cumulative": ([I, I, I, I], None),
    "mft_diff": ([I, I, I, I], None),
    "mft_rolling": ([I, I, I, I, I, I, I, I], None),
    "mft_expanding": ([I, I, I, I, I, I], None),
    "mft_group_reduce": ([I] * 10, None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_FEATURETOOLS_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_FEATURETOOLS_LIB does not exist: {LIB}")
    source = os.path.join(ROOT, "src", "kernels.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    script = os.path.join(ROOT, "build", "build.sh")
    proc = subprocess.run(
        ["bash", script], capture_output=True, text=True, timeout=1800
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(values, *, copy: bool = False) -> np.ndarray:
    original = np.asarray(values)
    if original.ndim != 1:
        raise ValueError("numeric input must be one-dimensional")
    if original.dtype.kind in "iu" and original.size:
        # Float64 cannot distinguish every integer outside this interval.
        # Reject those inputs instead of silently changing their values.
        limit = 1 << 53
        if np.any(original > limit) or (
            original.dtype.kind == "i" and np.any(original < -limit)
        ):
            raise ValueError("integer input cannot be represented exactly as float64")
    if original.dtype.kind == "f" and original.dtype.itemsize > 8:
        raise ValueError("floating-point input wider than float64 is not supported")
    if copy:
        return np.array(original, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(original, dtype=np.float64)


def i64(values) -> np.ndarray:
    original = np.asarray(values)
    if original.ndim != 1:
        raise ValueError("integer input must be one-dimensional")
    if original.dtype.kind not in "iu":
        raise TypeError("group codes must have an integer dtype")
    if original.dtype.kind == "u" and original.size:
        if np.any(original > np.iinfo(np.int64).max):
            raise ValueError("group code is outside the int64 range")
    return np.ascontiguousarray(original, dtype=np.int64)


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous NumPy arrays")
    address = int(array.ctypes.data)
    if not address:
        raise RuntimeError("NumPy returned a null buffer address")
    return address
