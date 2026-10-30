"""Cleaning for the two synthetic EV pack datasets.

Same rules as src/preprocessing/cleaning.py:
  * work within one battery / vehicle at a time;
  * anything that becomes a model input is causal (only that unit's past and present
    rows), so adding a later observation never changes an earlier row;
  * parameters learned from data (the outlier scales) are fitted on the training
    units only (`OutlierMasker.fit`) and then applied to every unit.

ev_degradation (one row per battery observation)
  1. text categories stripped; ids and targets must be present and unique
  2. `capacity_fade_pct` is 100 - SoH (checked): it is the target, so it is removed
     from the inputs and kept only as `leak_capacity_fade_pct` for the leakage check
  3. attributes that are fixed per battery (chemistry, maker, model, pack size,
     manufacture date, climate) are filled from the battery's other rows; conflicts
     raise. These are known for any real pack, so this is not future information
  4. age <-> dates: age = observation date - manufacture date, filled both ways
  5. physically impossible values -> NaN (PLAUSIBLE_RANGES)
  6. sensor outliers flagged (OutlierMasker: robust residual vs. the battery's recent
     readings; the first two readings of a battery are only range-checked). Flagged
     values are kept unless mask_outliers=True (see clean_ev_degradation)

ev_telematics (one row per vehicle trip)
  1. ids unique, no missing values, vehicle spec constant per vehicle (checked)
  2. impossible values -> NaN; SoC / SoH within 0-100
  3. `range_remaining_km` re-encodes SoH (range x Wh/km / (SoC x capacity) = SoH
     within +/-1 %, checked): renamed `leak_range_remaining_km`, never an input
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.cleaning import MAD_TO_STD, residual_vs_recent

EV1_STATIC = ["chemistry", "manufacturer", "vehicle_model", "pack_capacity_kwh",
              "manufacture_date", "climate_zone"]
EV1_CATEGORICAL = ["chemistry", "manufacturer", "vehicle_model", "climate_zone", "bms_firmware_version"]
EV1_TARGETS = ["state_of_health_pct", "remaining_useful_life_cycles", "failure_risk_label",
               "degradation_category"]

EV2_STATIC = ["manufacturer", "model", "year", "battery_capacity_kwh", "battery_type",
              "motor_power_kw", "vehicle_weight_kg"]
EV2_CATEGORICAL = ["manufacturer", "model", "battery_type", "weather", "road_type", "driving_style"]

# Physically possible ranges (inclusive). Values outside are sensor or entry errors.
PLAUSIBLE_RANGES = {
    # ev_degradation
    "pack_capacity_kwh": (5, 250), "battery_age_days": (0, 365 * 30), "cycle_count": (0, 20000),
    "avg_daily_distance_km": (0, 1500), "avg_charging_power_kw": (0.5, 400),
    "fast_charge_ratio": (0, 1), "avg_depth_of_discharge_pct": (0, 100),
    "avg_state_of_charge_pct": (0, 100), "charge_cycles_per_month": (0, 200),
    "total_energy_throughput_kwh": (0, 5e6), "avg_ambient_temp_c": (-50, 60),
    "humidity_pct": (0, 100), "altitude_m": (-500, 6000), "internal_resistance_mohm": (0.1, 1000),
    "voltage_avg_v": (100, 1000), "current_avg_a": (0, 1500), "charge_time_avg_min": (1, 24 * 60),
    "peak_temp_during_fast_charge_c": (-50, 90), "service_count": (0, 100),
    # ev_telematics
    "battery_capacity_kwh": (5, 250), "distance_km": (0, 1500), "avg_speed": (0, 250),
    "ambient_temp_c": (-50, 60), "battery_voltage_v": (100, 1000), "battery_current_a": (0, 1500),
    "battery_temp_c": (-50, 90), "soc_pct": (0, 100), "battery_cycles": (0, 20000),
    "charging_power_kw": (0.5, 400), "charging_duration_min": (0, 24 * 60),
    "energy_consumed_kwh": (0, 250),
}


def _strip_text(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA).astype(object)
        df[c] = df[c].where(df[c].notna(), np.nan)


def apply_plausible_ranges(df: pd.DataFrame, report: dict) -> None:
    for c, (lo, hi) in PLAUSIBLE_RANGES.items():
        if c in df:
            bad = df[c].notna() & ~df[c].between(lo, hi)
            report[f"implausible_{c}"] = int(bad.sum())
            df.loc[bad, c] = np.nan


def fill_static(df: pd.DataFrame, unit: str, cols: list[str], report: dict) -> None:
    """Fill per-unit constant attributes from the unit's other rows; fail on conflicts."""
    g = df.groupby(unit, sort=False)
    for c in cols:
        n_distinct = g[c].nunique()
        if (n_distinct > 1).any():
            raise ValueError(f"{c} is not constant within {int((n_distinct > 1).sum())} {unit}s")
        before = int(df[c].isna().sum())
        df[c] = g[c].transform("first")          # "first" skips NaN
        report[f"filled_static_{c}"] = before - int(df[c].isna().sum())


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #
class OutlierMasker:
    """Causal robust outlier check on sensor columns, applied per unit (battery).

    For each reading: residual = value - median of the unit's previous `window`
    readings (log scale for strictly positive quantities, where faults are
    multiplicative). A quantity that rises with age (resistance) gives positive
    residuals even without faults, so `fit` learns the typical residual (median) and
    its robust spread (MAD) per column on the training units, and `flags` marks
    |residual - typical| > n_sigma * spread. Readings without at least two earlier
    readings are left to the plausibility check.

    Known limitation: a spike stays in the next `window` readings' reference and shifts
    its median by one rank, which can flag a normal reading after it. Leaving flagged
    readings out of the reference was tried and lowered precision on ev_degradation
    (0.77 -> 0.73, same recall), because it also turns genuine level shifts into runs.
    """

    def __init__(self, columns: list[str], log_columns: set[str], unit: str = "battery_id",
                 window: int = 5, n_sigma: float = 4.0):
        self.columns, self.log_columns, self.unit = columns, set(log_columns), unit
        self.window, self.n_sigma = window, n_sigma
        self.center_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}

    def _values(self, df: pd.DataFrame, c: str) -> np.ndarray:
        x = df[c].to_numpy(float)
        if c in self.log_columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                x = np.where(x > 0, np.log(x), np.nan)
        return x

    def _residuals(self, df: pd.DataFrame, c: str) -> np.ndarray:
        """Residual vs the median of the unit's previous `window` readings (all of them:
        excluding flagged ones would make every reading after a genuine level shift
        an outlier; a median of 5 is not moved by a single bad reading)."""
        x = self._values(df, c)
        resid = np.full(len(x), np.nan)
        for idx in df.groupby(self.unit, sort=False).indices.values():
            v = pd.Series(x[idx])
            n_prev = v.notna().astype(int).shift(1).rolling(self.window, min_periods=1).sum()
            r = residual_vs_recent(v.to_numpy(), self.window)
            resid[idx] = np.where(n_prev >= 2, r, np.nan)
        return resid

    def fit(self, df: pd.DataFrame) -> "OutlierMasker":
        for c in self.columns:
            r = self._residuals(df, c)
            r = r[np.isfinite(r)]
            self.center_[c] = float(np.median(r))
            self.scale_[c] = float(MAD_TO_STD * np.median(np.abs(r - self.center_[c])))
        return self

    def flags(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.scale_:
            raise RuntimeError("OutlierMasker.fit must be called first")
        return pd.DataFrame({c: np.abs(self._residuals(df, c) - self.center_[c]) > self.n_sigma * self.scale_[c]
                             for c in self.columns}, index=df.index)

    def to_dict(self) -> dict:
        return {"columns": self.columns, "log_columns": sorted(self.log_columns), "unit": self.unit,
                "window": self.window, "n_sigma": self.n_sigma, "center": self.center_, "scale": self.scale_}

    @classmethod
    def from_dict(cls, d: dict) -> "OutlierMasker":
        m = cls(d["columns"], set(d["log_columns"]), d["unit"], d["window"], d["n_sigma"])
        m.center_, m.scale_ = dict(d["center"]), dict(d["scale"])
        return m


EV1_OUTLIER_COLUMNS = ["avg_charging_power_kw", "avg_ambient_temp_c", "internal_resistance_mohm",
                       "current_avg_a", "charge_time_avg_min", "peak_temp_during_fast_charge_c"]
EV1_LOG_COLUMNS = {"avg_charging_power_kw", "internal_resistance_mohm", "current_avg_a",
                   "charge_time_avg_min"}


def make_ev1_masker() -> OutlierMasker:
    return OutlierMasker(EV1_OUTLIER_COLUMNS, EV1_LOG_COLUMNS, unit="battery_id")


# --------------------------------------------------------------------------- #
# Dataset-level cleaning
# --------------------------------------------------------------------------- #
def clean_ev_degradation(df: pd.DataFrame, masker: OutlierMasker | None = None,
                         mask_outliers: bool = False,
                         require_targets: bool = True) -> tuple[pd.DataFrame, dict]:
    """Steps 1-6 of the module docstring. `masker` must already be fitted (or None, e.g.
    to fit it on the range-checked data). With a masker, every row gets
    `flag_outlier_<column>` and `n_outlier_flags`; the values are set to NaN only with
    `mask_outliers=True`. Default False: on the development batteries masking made no
    measurable difference to the SoH error (MAE 0.589 vs 0.590, README EV section),
    and the detector finds only ~21 % of the injected outliers while ~23 % of its
    flags are genuine readings, so blanking would discard real data for no gain."""
    df = df.copy()
    rep: dict = {"rows_in": len(df)}
    if df.battery_id.isna().any() or df.observation_id.isna().any():
        raise ValueError("battery_id and observation_id must be present in every row")
    if df.observation_id.duplicated().any():
        raise ValueError("duplicate observation_id")
    if require_targets and df[EV1_TARGETS].isna().any().any():
        raise ValueError("targets have missing values")
    _strip_text(df, [c for c in EV1_CATEGORICAL if c in df])
    if "obs_seq" not in df:
        df["obs_seq"] = df.observation_id.str.extract(r"(\d+)$", expand=False).astype(int)
    df = df.sort_values(["battery_id", "obs_seq"]).reset_index(drop=True)

    if "capacity_fade_pct" in df:
        if "state_of_health_pct" in df:
            both = df[["capacity_fade_pct", "state_of_health_pct"]].dropna()
            rep["capacity_fade_equals_100_minus_soh"] = bool(
                np.allclose(both.capacity_fade_pct, 100 - both.state_of_health_pct, atol=0.006))
        df = df.rename(columns={"capacity_fade_pct": "leak_capacity_fade_pct"})

    fill_static(df, "battery_id", EV1_STATIC, rep)

    from_dates = (df.observation_date - df.manufacture_date).dt.days
    known = df.battery_age_days.notna() & from_dates.notna()
    rep["age_matches_dates"] = bool((df.battery_age_days[known] == from_dates[known]).all())
    n0 = int(df.battery_age_days.isna().sum())
    df["battery_age_days"] = df.battery_age_days.fillna(from_dates)
    df["observation_date"] = df.observation_date.fillna(
        df.manufacture_date + pd.to_timedelta(df.battery_age_days, unit="D"))
    rep["filled_battery_age_days"] = n0 - int(df.battery_age_days.isna().sum())
    age_steps = df.groupby("battery_id").battery_age_days.diff()
    rep["age_decreasing_steps"] = int((age_steps < 0).sum())   # 0 = id order is time order

    apply_plausible_ranges(df, rep)
    if masker is not None:
        fl = masker.flags(df)
        for c in masker.columns:
            df[f"flag_outlier_{c}"] = fl[c].to_numpy()
            rep[f"outlier_flagged_{c}"] = int(fl[c].sum())
            if mask_outliers:
                df.loc[fl[c].to_numpy(), c] = np.nan
        df["n_outlier_flags"] = fl.sum(axis=1).to_numpy()
        rep["outliers_masked"] = bool(mask_outliers)
    rep["rows_out"] = len(df)
    return df, rep


def clean_ev_telematics(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    rep: dict = {"rows_in": len(df)}
    if df.trip_id.duplicated().any():
        raise ValueError("duplicate trip_id")
    _strip_text(df, [c for c in EV2_CATEGORICAL if c in df])
    rep["missing_values"] = int(df.isna().sum().sum())
    fill_static(df, "vehicle_id", EV2_STATIC, rep)
    if "range_remaining_km" in df:
        if "soh_pct" in df:
            implied = (df.range_remaining_km * df.energy_consumed_kwh / df.distance_km
                       / (df.soc_pct / 100 * df.battery_capacity_kwh) / (df.soh_pct / 100))
            rep["range_implied_soh_ratio_p01_p99"] = [round(float(implied.quantile(q)), 4) for q in (0.01, 0.99)]
        df = df.rename(columns={"range_remaining_km": "leak_range_remaining_km"})
    apply_plausible_ranges(df, rep)
    if "soh_pct" in df:
        bad = ~df.soh_pct.between(0, 100)
        rep["implausible_soh_pct"] = int(bad.sum())
        df = df[~bad]
    rep["same_day_trips"] = int(df.duplicated(["vehicle_id", "trip_date"]).sum())
    if "trip_seq" not in df:
        df["trip_seq"] = df.trip_id.str.extract(r"(\d+)$", expand=False).astype(int)
    df = df.sort_values(["vehicle_id", "trip_date", "trip_seq"]).reset_index(drop=True)
    rep["rows_out"] = len(df)
    return df, rep
