import numpy as np
import pandas as pd

from src.preprocessing.cleaning import (hampel_flags, interpolate_short_gaps, moving_average,
                                        nan_run_lengths, noise_std, causal_rolling_slope)


def test_nan_run_lengths():
    x = np.array([1, np.nan, np.nan, 4, np.nan, 6.0])
    assert nan_run_lengths(x).tolist() == [0, 2, 2, 0, 1, 0]


def test_interpolates_only_short_interior_gaps():
    t = np.arange(10.0)
    x = np.array([0, 1, np.nan, 3, np.nan, np.nan, np.nan, np.nan, 8, 9.0])
    out = interpolate_short_gaps(t, x, max_gap=2)
    assert out[2] == 2.0                      # 1-sample gap filled
    assert np.isnan(out[4:8]).all()           # 4-sample gap left alone


def test_edge_gaps_not_extrapolated():
    t = np.arange(5.0)
    out = interpolate_short_gaps(t, np.array([np.nan, 1, 2, 3, np.nan]), max_gap=3)
    assert np.isnan(out[0]) and np.isnan(out[-1])


def test_moving_average_is_causal():
    x = np.array([0, 0, 0, 100.0, 0, 0])
    ma = moving_average(x, 3, causal=True)
    assert ma[:3].tolist() == [0, 0, 0]       # the spike at index 3 cannot leak backwards
    assert ma[3] > 0


def test_hampel_flags_spike_not_trend():
    x = np.linspace(2.0, 1.0, 60)
    x[30] = 0.3
    flags = hampel_flags(x, window=7, n_sigma=3.0, min_scale=0.02)
    assert flags[30] and flags.sum() == 1


def test_noise_std_recovers_white_noise_on_a_trend():
    rng = np.random.default_rng(0)
    x = np.linspace(4, 3, 5000) + rng.normal(0, 0.01, 5000)
    assert abs(noise_std(x) - 0.01) < 0.002


def test_causal_slope_uses_only_past():
    y = np.r_[np.zeros(10), np.arange(10.0)]
    s = causal_rolling_slope(y, window=5)
    assert np.isnan(s[:2]).all()              # fewer than 3 points -> undefined
    assert np.allclose(s[2:11], 0)            # unaffected by the later ramp


def test_residual_vs_recent_is_causal_and_matches_definition():
    import numpy as np
    from src.preprocessing.cleaning import residual_vs_recent
    x = np.array([1.0, 2.0, 4.0, 3.0, 10.0, 2.0])
    r = residual_vs_recent(x, 3)
    assert np.isnan(r[0])
    assert r[3] == 3.0 - np.median([1.0, 2.0, 4.0])
    assert r[4] == 10.0 - np.median([2.0, 4.0, 3.0])
    np.testing.assert_array_equal(residual_vs_recent(x[:4], 3), r[:4])
