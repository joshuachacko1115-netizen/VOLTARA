"""Feature engineering for the synthetic EV pack datasets.

All per-unit history features are causal: row t of a battery / vehicle only uses
rows 0..t of the same unit (expanding or trailing windows, shift(1) for "previous").
No feature uses another unit's rows or any target, so features can be built once
for all rows before the battery-wise split. tests/test_ev_pipeline.py checks that
appending later rows never changes earlier features.

Inputs that are deliberately NOT features (see README, EV section):
  ev_degradation  leak_capacity_fade_pct (= 100 - SoH), the four targets
  ev_telematics   leak_range_remaining_km (re-encodes SoH), soh_pct, the risk labels;
                  for the consumption model also energy, current and battery
                  temperature, which are computed from the trip's energy use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.ev_cleaning import EV1_CATEGORICAL, EV2_CATEGORICAL

EV1_RAW_NUMERIC = ["pack_capacity_kwh", "battery_age_days", "cycle_count", "avg_daily_distance_km",
                   "avg_charging_power_kw", "fast_charge_ratio", "avg_depth_of_discharge_pct",
                   "avg_state_of_charge_pct", "charge_cycles_per_month", "total_energy_throughput_kwh",
                   "avg_ambient_temp_c", "humidity_pct", "altitude_m", "internal_resistance_mohm",
                   "voltage_avg_v", "current_avg_a", "charge_time_avg_min",
                   "peak_temp_during_fast_charge_c", "service_count", "recall_flag"]
# Usage / environment readings that describe how hard the battery is driven. Their
# running mean over the battery's history is a less noisy estimate of its stress level.
EV1_STRESS = ["avg_ambient_temp_c", "avg_depth_of_discharge_pct", "fast_charge_ratio",
              "avg_state_of_charge_pct", "avg_charging_power_kw", "charge_cycles_per_month",
              "avg_daily_distance_km", "humidity_pct", "altitude_m", "peak_temp_during_fast_charge_c"]

EV2_RAW_NUMERIC = ["year", "battery_capacity_kwh", "motor_power_kw", "vehicle_weight_kg", "distance_km",
                   "avg_speed", "ambient_temp_c", "battery_voltage_v", "battery_current_a",
                   "battery_temp_c", "soc_pct", "battery_cycles", "charging_power_kw",
                   "charging_duration_min", "energy_consumed_kwh"]
# Known before a trip starts: used for the energy-consumption (Wh/km) model.
EV2_PRETRIP_NUMERIC = ["year", "battery_capacity_kwh", "motor_power_kw", "vehicle_weight_kg",
                       "distance_km", "avg_speed", "ambient_temp_c", "battery_cycles"]
EV2_PRETRIP_CATEGORICAL = ["manufacturer", "model", "battery_type", "weather", "road_type", "driving_style"]
DC_FAST_KW = 22.0   # above the highest AC rate in the data (22 kW): DC fast charging


def _expanding(g, col: str, how: str) -> pd.Series:
    return g[col].transform(lambda s: getattr(s.expanding(), how)())


def causal_initial_nan(values: np.ndarray, n: int = 3) -> np.ndarray:
    """Median of the first n *available* values seen so far (NaN until one exists)."""
    out = np.full(len(values), np.nan)
    seen: list[float] = []
    for k, v in enumerate(values):
        if np.isfinite(v) and len(seen) < n:
            seen.append(v)
        if seen:
            out[k] = np.median(seen)
    return out


def causal_slope(x: np.ndarray, y: np.ndarray, min_points: int = 3) -> np.ndarray:
    """Least-squares slope dy/dx over all rows so far (inclusive)."""
    out = np.full(len(x), np.nan)
    for k in range(len(x)):
        m = np.isfinite(x[:k + 1]) & np.isfinite(y[:k + 1])
        if m.sum() >= min_points and np.ptp(x[:k + 1][m]) > 0:
            out[k] = np.polyfit(x[:k + 1][m], y[:k + 1][m], 1)[0]
    return out


def ev1_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add engineered features to cleaned ev_degradation rows (sorted by battery, time)."""
    df = df.copy()
    new = []
    df["efc"] = df.total_energy_throughput_kwh / df.pack_capacity_kwh       # equivalent full cycles
    df["cycles_per_day"] = df.cycle_count / df.battery_age_days.clip(lower=1)
    df["efc_per_day"] = df.efc / df.battery_age_days.clip(lower=1)
    since = (df.observation_date - df.last_service_date).dt.days
    since = since.where(since >= 0)
    # Never serviced -> time since new. A missing date with service_count > 0 stays NaN.
    df["days_since_service"] = np.where(df.service_count == 0, df.battery_age_days, since)
    new += ["efc", "cycles_per_day", "efc_per_day", "days_since_service"]

    g = df.groupby("battery_id", sort=False)
    df["n_obs_so_far"] = g.cumcount() + 1
    df["first_age_days"] = _expanding(g, "battery_age_days", "min")
    new += ["n_obs_so_far", "first_age_days"]
    for c in EV1_STRESS:
        df[f"{c}_hist_mean"] = _expanding(g, c, "mean")
        new.append(f"{c}_hist_mean")
    for c in ["internal_resistance_mohm", "cycle_count", "efc"]:
        df[f"{c}_med3"] = g[c].transform(lambda s: s.rolling(3, min_periods=1).median())
        df[f"{c}_prev"] = g[c].transform(lambda s: s.ffill().shift(1))
        new += [f"{c}_med3", f"{c}_prev"]
    df["r_initial_mohm"] = g.internal_resistance_mohm.transform(
        lambda s: pd.Series(causal_initial_nan(s.to_numpy(float)), index=s.index))
    df["r_growth"] = df.internal_resistance_mohm_med3 / df.r_initial_mohm
    slopes = pd.Series(np.nan, index=df.index)
    for _, idx in g.indices.items():
        sub = df.iloc[idx]
        slopes.iloc[idx] = 1000 * causal_slope(sub.cycle_count.to_numpy(float),
                                               sub.internal_resistance_mohm.to_numpy(float))
    df["r_slope_per_1000_cycles"] = slopes
    new += ["r_initial_mohm", "r_growth", "r_slope_per_1000_cycles"]
    return df, EV1_RAW_NUMERIC + new


def ev2_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add engineered features to cleaned ev_telematics rows (sorted by vehicle, time)."""
    df = df.copy()
    new = []
    # Purchase date is unknown: mid-model-year is used as the reference point.
    df["vehicle_age_years"] = (df.trip_date - pd.to_datetime(df.year.astype(int).astype(str) + "-07-01")
                               ).dt.days / 365.25
    df["cycles_per_year"] = df.battery_cycles / df.vehicle_age_years.clip(lower=0.1)
    df["wh_per_km"] = df.energy_consumed_kwh / df.distance_km * 1000
    df["batt_temp_rise_c"] = df.battery_temp_c - df.ambient_temp_c
    df["discharge_c_rate"] = df.battery_current_a * df.battery_voltage_v / 1000 / df.battery_capacity_kwh
    df["charge_c_rate"] = df.charging_power_kw / df.battery_capacity_kwh
    df["dc_fast_charge"] = (df.charging_power_kw > DC_FAST_KW).astype(float)
    df["trip_energy_frac"] = df.energy_consumed_kwh / df.battery_capacity_kwh
    df["trip_date_ord"] = (df.trip_date - pd.Timestamp("2019-01-01")).dt.days
    new += ["vehicle_age_years", "cycles_per_year", "wh_per_km", "batt_temp_rise_c", "discharge_c_rate",
            "charge_c_rate", "dc_fast_charge", "trip_energy_frac", "trip_date_ord"]

    g = df.groupby("vehicle_id", sort=False)
    df["n_trips_so_far"] = g.cumcount() + 1
    for c in ["battery_temp_c", "dc_fast_charge", "discharge_c_rate", "charge_c_rate"]:
        df[f"{c}_hist_mean"] = _expanding(g, c, "mean")
        new.append(f"{c}_hist_mean")
    df["battery_temp_hist_max"] = _expanding(g, "battery_temp_c", "max")
    df["aggressive_share_so_far"] = (df.driving_style == "Aggressive").astype(float).groupby(
        df.vehicle_id, sort=False).transform(lambda s: s.expanding().mean())
    df["cycles_prev"] = g.battery_cycles.shift(1)
    df["cycles_hist_max"] = _expanding(g, "battery_cycles", "max")
    df["days_since_prev_trip"] = g.trip_date.diff().dt.days
    new += ["n_trips_so_far", "battery_temp_hist_max", "aggressive_share_so_far", "cycles_prev",
            "cycles_hist_max", "days_since_prev_trip"]
    return df, EV2_RAW_NUMERIC + new


def implied_soh_from_range(df: pd.DataFrame) -> pd.Series:
    """SoH (%) re-derived from the range estimate. Leakage check only, never a feature."""
    return (df.leak_range_remaining_km * df.energy_consumed_kwh / df.distance_km
            / (df.soc_pct / 100 * df.battery_capacity_kwh) * 100)


EV1_CATEGORICAL_FEATURES = EV1_CATEGORICAL
EV2_CATEGORICAL_FEATURES = EV2_CATEGORICAL
