"""Build the modelling tables from the interim parquet files.

Outputs (data/processed/)
  ba_cycles.parquet   Battery Aging, one row per discharge cycle: features, quality
                      flags, SoH, EOL and RUL targets
  rw_checks.parquet   Randomized Usage, one row per reference check: same idea
  reports/data_quality_ba.csv, reports/data_quality_rw.csv   per-cell summaries

Usage:
    python -m src.features.build_features
"""
from __future__ import annotations

import pandas as pd

from src.config import INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR
from src.features.battery_aging import build_cycle_table
from src.features.randomized_usage import build_check_table
from src.features.targets import (BA_FLAG_COLUMNS, ba_quality_flags, ba_targets_and_trends,
                                  rw_targets_and_trends)


def build_ba() -> pd.DataFrame:
    ops = pd.read_parquet(INTERIM_DIR / "ba_operations.parquet")
    ts = pd.read_parquet(INTERIM_DIR / "ba_timeseries.parquet",
                         columns=["cell", "op_index", "type", "time_s", "voltage_measured",
                                  "current_measured", "temperature_measured"])
    cycles = build_cycle_table(ops, ts)
    cycles = ba_targets_and_trends(ba_quality_flags(cycles))
    cycles.to_parquet(PROCESSED_DIR / "ba_cycles.parquet", index=False)

    q = cycles.groupby("cell").agg(
        n_cycles=("cycle_number", "size"),
        **{f: (f, "sum") for f in BA_FLAG_COLUMNS},
        n_valid=("is_valid", "sum"),
        soh_first=("soh", lambda s: s[cycles.loc[s.index, "is_valid"]].iloc[0]
                   if cycles.loc[s.index, "is_valid"].any() else float("nan")),
        soh_last=("soh", lambda s: s[cycles.loc[s.index, "is_valid"]].iloc[-1]
                  if cycles.loc[s.index, "is_valid"].any() else float("nan")),
        eol_cycle=("eol_cycle", "first"),
        in_model_cohort=("in_model_cohort", "first"),
    )
    q.to_csv(REPORTS_DIR / "data_quality_ba.csv")
    print("Battery Aging data quality\n", q.to_string())
    return cycles


def build_rw() -> pd.DataFrame:
    steps = pd.read_parquet(INTERIM_DIR / "rw_steps.parquet")
    charges = pd.read_parquet(INTERIM_DIR / "rw_timeseries.parquet",
                              filters=[("comment", "==", "reference charge")])
    checks = rw_targets_and_trends(build_check_table(steps, charges))
    checks.to_parquet(PROCESSED_DIR / "rw_checks.parquet", index=False)

    bad_t = steps.groupby("cell").n_bad_temperature.sum()
    q = checks.groupby("cell", sort=False).agg(
        group=("group", "first"), n_checks=("check_id", "size"), n_valid=("is_valid", "sum"),
        capacity_initial_ah=("capacity_initial_ah", "first"),
        soh_last=("soh", "last"), efc_last=("efc", "max"), eol_efc=("eol_efc", "first"))
    q["n_bad_temperature_samples"] = bad_t.reindex(q.index)
    q.to_csv(REPORTS_DIR / "data_quality_rw.csv")
    print("\nRandomized Usage data quality\n", q.to_string())
    return checks


def run() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    build_ba()
    build_rw()


if __name__ == "__main__":
    run()
