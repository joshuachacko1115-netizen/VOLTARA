"""Find and load the two synthetic EV pack datasets in data/raw/.

The CSVs are identified by their header (EV_DATASETS[...]["signature"]), so the
folder and file names do not matter. The raw files are only read, never modified.

* ev_degradation  one row per (battery pack, observation); 35 columns.
* ev_telematics   one row per (vehicle, trip + following charge); 30 columns.
                  Column names are converted to snake_case on load.

`injected_outlier_oracle` reads a generator artefact (see the function) that marks
the outliers injected into ev_degradation. It is used only to *audit* the outlier
detector in src/preprocessing/ev_cleaning.py, never to clean or as a feature.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import EV_DATASETS, RAW_DIR

EV1_DATE_COLS = ["manufacture_date", "observation_date", "last_service_date"]
EV1_SENSOR_COLS = ["avg_charging_power_kw", "avg_ambient_temp_c", "internal_resistance_mohm",
                   "current_avg_a", "charge_time_avg_min", "peak_temp_during_fast_charge_c"]


def snake(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")


def find_ev_files(root: Path = RAW_DIR) -> dict[str, Path]:
    """Map dataset key -> CSV path by matching each CSV's header to a signature."""
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*.csv")):
        try:
            header = set(pd.read_csv(path, nrows=0).columns.str.strip())
        except Exception:
            continue
        for key, meta in EV_DATASETS.items():
            if meta["signature"] <= header:
                if key in found:
                    raise ValueError(f"Two files look like {key}: {found[key]} and {path}")
                found[key] = path
    return found


def load_ev_degradation(path: Path | None = None) -> pd.DataFrame:
    path = path or find_ev_files()["ev_degradation"]
    df = pd.read_csv(path, parse_dates=EV1_DATE_COLS, float_precision="round_trip")
    # Observation ids increase with time inside every battery (checked in ev_cleaning),
    # so the numeric part is used as the time order when dates or ages are missing.
    df["obs_seq"] = df.observation_id.str.extract(r"(\d+)$", expand=False).astype(int)
    return df.sort_values(["battery_id", "obs_seq"]).reset_index(drop=True)


def load_ev_telematics(path: Path | None = None) -> pd.DataFrame:
    path = path or find_ev_files()["ev_telematics"]
    df = pd.read_csv(path, float_precision="round_trip")
    df.columns = [snake(c) for c in df.columns]
    df["trip_date"] = pd.to_datetime(df.trip_date)
    df["trip_seq"] = df.trip_id.str.extract(r"(\d+)$", expand=False).astype(int)
    return df.sort_values(["vehicle_id", "trip_date", "trip_seq"]).reset_index(drop=True)


def injected_outlier_oracle(path: Path | None = None) -> pd.DataFrame:
    """Boolean frame (same row order as load_ev_degradation) marking injected outliers.

    The generator rounds normal values to 1-2 decimals but writes the injected
    outliers unrounded (11-17 decimals). That is an artefact of this synthetic file,
    so it is used only to measure how well the general-purpose detector works.
    """
    path = path or find_ev_files()["ev_degradation"]
    raw = pd.read_csv(path, dtype=str)
    raw["obs_seq"] = raw.observation_id.str.extract(r"(\d+)$", expand=False).astype(int)
    raw = raw.sort_values(["battery_id", "obs_seq"]).reset_index(drop=True)
    return pd.DataFrame({c: raw[c].str.split(".").str[1].str.len().fillna(0).ge(6)
                         for c in EV1_SENSOR_COLS})
