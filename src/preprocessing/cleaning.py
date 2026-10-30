"""Signal- and cycle-level cleaning utilities.

Everything here works *within a single cell* and never looks at other cells, so
applying it before a leave-cells-out split cannot leak information between the
training and test cells. Steps that learn parameters from data (imputation,
scaling) are NOT done here; they live inside the sklearn Pipelines in
src/models and are fitted on training folds only.

Two kinds of operation are kept apart on purpose:
  * causal operations (only past and present values) - safe for model features;
  * centred operations (use neighbours on both sides) - used only for data-quality
    flags and for defining targets such as the end-of-life cycle, never as inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAD_TO_STD = 1.4826  # MAD -> standard deviation for normally distributed noise


# --------------------------------------------------------------------------- #
# Gaps and smoothing for sampled signals
# --------------------------------------------------------------------------- #
def nan_run_lengths(x: np.ndarray) -> np.ndarray:
    """Length of the NaN run each element belongs to (0 for finite values)."""
    isnan = ~np.isfinite(np.asarray(x, dtype=float))
    out = np.zeros(isnan.size, dtype=int)
    i = 0
    while i < isnan.size:
        if isnan[i]:
            j = i
            while j < isnan.size and isnan[j]:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


def interpolate_short_gaps(t: np.ndarray, x: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Linearly interpolate interior NaN runs of at most `max_gap` samples.

    Longer runs and gaps at either end stay NaN - interpolating across them would
    invent data, so they are left for the caller to drop or flag.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float).copy()
    good = np.isfinite(x)
    if good.sum() < 2 or good.all():
        return x
    runs = nan_run_lengths(x)
    first, last = np.argmax(good), len(x) - 1 - np.argmax(good[::-1])
    fill = (~good) & (runs <= max_gap)
    fill[:first] = False
    fill[last + 1:] = False
    x[fill] = np.interp(t[fill], t[good], x[good])
    return x


def moving_average(x, window: int, causal: bool = True) -> np.ndarray:
    """Moving average. Causal (trailing window) by default so it is safe as a feature."""
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rolling(window, min_periods=1, center=not causal).mean().to_numpy()


def noise_std(x: np.ndarray) -> float:
    """Robust estimate of white measurement noise from second differences.

    Second differences cancel any locally linear trend (a discharge curve), and for
    white noise var(x[i-1] - 2x[i] + x[i+1]) = 6 sigma^2. The MAD makes the estimate
    insensitive to genuine steps such as load changes.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return np.nan
    d2 = np.diff(x, n=2)
    return float(MAD_TO_STD * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6))


# --------------------------------------------------------------------------- #
# Cycle-level outlier flags
# --------------------------------------------------------------------------- #
def hampel_flags(x, window: int = 7, n_sigma: float = 3.0, min_scale: float = 0.0) -> np.ndarray:
    """Flag points that deviate from the centred rolling median by > n_sigma robust std.

    Centred, so it is a quality flag only. `min_scale` stops perfectly flat stretches
    (MAD = 0) from flagging tiny wiggles.
    """
    s = pd.Series(np.asarray(x, dtype=float))
    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
    scale = np.maximum(MAD_TO_STD * mad, min_scale)
    return ((s - med).abs() > n_sigma * scale).fillna(False).to_numpy()


def causal_rolling_slope(y, x=None, window: int = 10, min_points: int = 3) -> np.ndarray:
    """Least-squares slope dy/dx over the trailing `window` valid points (inclusive)."""
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, dtype=float)
    out = np.full(len(y), np.nan)
    valid_idx = np.flatnonzero(np.isfinite(y) & np.isfinite(x))
    for k, i in enumerate(valid_idx):
        idx = valid_idx[max(0, k - window + 1): k + 1]
        if len(idx) >= min_points and np.ptp(x[idx]) > 0:
            out[i] = np.polyfit(x[idx], y[idx], 1)[0]
    return out


def causal_rolling(values, window: int, how: str = "median") -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=float))
    r = s.rolling(window, min_periods=1)
    return getattr(r, how)().to_numpy()


def causal_initial(values, n: int = 3) -> np.ndarray:
    """Running estimate of the initial value: median of the first min(t + 1, n) points.

    The full-history median of the first n points would let rows 0 .. n-2 see later cycles.
    """
    v = np.asarray(values, dtype=float)
    return np.array([np.median(v[:min(k + 1, n)]) for k in range(len(v))])


def residual_vs_recent(values, window: int) -> np.ndarray:
    """Value minus the median of the previous `window` values (causal; NaN for the first row)."""
    s = pd.Series(np.asarray(values, dtype=float))
    return (s - s.shift(1).rolling(window, min_periods=1).median()).to_numpy()
