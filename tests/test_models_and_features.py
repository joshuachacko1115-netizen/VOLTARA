import numpy as np
import pandas as pd

from src.features.battery_aging import discharge_features
from src.models.common import cross_validate, make_pipeline
from src.evaluation.splits import leave_cells_out
from sklearn.linear_model import LinearRegression
from src.models.range_estimation import RangeAssumptions, estimate_range_km


def _fake_discharge(cap_ah=1.8, i=2.0, n=400):
    """Constant-current discharge from 4.2 V to below 2.7 V."""
    t = np.linspace(0, cap_ah / i * 3600 * 1.01, n)
    v = np.linspace(4.2, 2.6, n)
    return pd.DataFrame({"time_s": t, "voltage_measured": v, "current_measured": -i,
                         "temperature_measured": 25.0 + t / t[-1] * 10})


def test_discharge_capacity_matches_coulomb_counting():
    f = discharge_features(_fake_discharge(1.8))
    # The fake voltage falls linearly from 4.2 V to 2.6 V over 1.01x the nominal
    # discharge time, so it reaches 2.7 V after (4.2 - 2.7) / (4.2 - 2.6) of that time.
    expected = 1.8 * 1.01 * (4.2 - 2.7) / (4.2 - 2.6)
    assert f["reached_cutoff"] and abs(f["capacity_ah_2v7"] - expected) < 0.01
    assert abs(f["i_set"] - 2.0) < 1e-6


def test_short_gap_in_discharge_is_interpolated():
    d = _fake_discharge()
    d.loc[100:101, "voltage_measured"] = np.nan
    f = discharge_features(d)
    assert f["n_dropped_samples"] == 0


def test_scaler_and_imputer_are_fitted_on_training_cells_only():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"cell": np.repeat(list("abc"), 20), "x": rng.normal(size=60)})
    df["y"] = 2 * df.x
    df.loc[df.cell == "c", "x"] = 1000.0            # extreme held-out cell
    train = df[df.cell != "c"]
    pipe = make_pipeline(LinearRegression()).fit(train[["x"]], train.y)
    assert abs(pipe.named_steps["scale"].mean_[0] - train.x.mean()) < 1e-9
    assert pipe.named_steps["scale"].mean_[0] < 5   # not pulled towards the held-out 1000s

    # The same must hold inside cross_validate: in the fold that holds out "c",
    # the scaled training data never contains anything near 1000.
    seen = {}

    class Spy(LinearRegression):
        def fit(self, X, y):
            seen[len(seen)] = np.abs(np.asarray(X)).max()
            return super().fit(X, y)

    cross_validate(df, ["x"], "y", leave_cells_out(df), "cells", {"spy": Spy()})
    assert seen[2] < 5


def test_range_scales_linearly_with_energy():
    a = RangeAssumptions()
    r = estimate_range_km([10.0, 8.0], a)
    assert r[1] / r[0] == np.float64(0.8) or abs(r[1] / r[0] - 0.8) < 1e-12
