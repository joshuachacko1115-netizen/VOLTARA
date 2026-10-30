"""Convert the extracted .mat files into parquet tables in data/processed/interim/.

Outputs
  ba_operations.parquet    Battery Aging: one row per charge/discharge/impedance operation
  ba_timeseries.parquet    Battery Aging: raw charge and discharge samples
  rw_steps.parquet         Randomized Usage: one row per step (all ~1.1 M steps, summarised)
  rw_timeseries.parquet    Randomized Usage: raw samples of reference / pulsed steps only
  cells.csv                one row per cell with its group and experiment metadata

Cells that appear in two groups (B0025-B0028) are loaded once, after checking that
the copies are byte-identical (MD5 in the extraction manifest).

Usage:
    python -m src.data.build_dataset
"""
from __future__ import annotations

import re
import time

import pandas as pd

from src.config import EXTRACTED_DIR, INTERIM_DIR
from src.data import extract
from src.data.loaders import load_battery_aging_cell, load_randomized_usage_cell


def _cell_sort_key(name: str) -> tuple:
    m = re.match(r"([A-Z]+)0*(\d+)", name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)


def unique_mat_files(manifest: pd.DataFrame, dataset: str) -> pd.DataFrame:
    mats = manifest[(manifest.dataset == dataset) & manifest.file.str.endswith(".mat")].copy()
    mats["cell"] = mats.file.str[:-4]
    if (mats.groupby("cell").md5.nunique() > 1).any():
        bad = mats.groupby("cell").md5.nunique()
        raise ValueError(f"Conflicting copies for cells: {list(bad[bad > 1].index)}")
    # Keep the copy from the group whose name is shortest/most specific is irrelevant
    # (identical bytes), so take the first one in manifest order.
    mats = mats.drop_duplicates("cell", keep="first")
    return mats.sort_values("cell", key=lambda s: s.map(_cell_sort_key)).reset_index(drop=True)


def build_battery_aging(manifest: pd.DataFrame) -> list[dict]:
    ops, series, meta = [], [], []
    for row in unique_mat_files(manifest, "battery_aging").itertuples():
        t0 = time.time()
        out = load_battery_aging_cell(EXTRACTED_DIR / row.relative_path)
        out["operations"]["group"] = row.group
        ops.append(out["operations"])
        series.append(out["timeseries"])
        o = out["operations"]
        meta.append({"dataset": "battery_aging", "cell": row.cell, "group": row.group,
                     "n_operations": len(o), "n_discharge": int((o.type == "discharge").sum()),
                     "ambient_temperatures": ",".join(str(x) for x in sorted(o.ambient_temperature.unique()))})
        print(f"  {row.cell}: {len(o)} operations ({time.time() - t0:.1f}s)")
    pd.concat(ops, ignore_index=True).to_parquet(INTERIM_DIR / "ba_operations.parquet", index=False)
    pd.concat(series, ignore_index=True).to_parquet(INTERIM_DIR / "ba_timeseries.parquet", index=False)
    return meta


def build_randomized_usage(manifest: pd.DataFrame) -> list[dict]:
    steps, series, meta = [], [], []
    for row in unique_mat_files(manifest, "randomized_usage").itertuples():
        t0 = time.time()
        out = load_randomized_usage_cell(EXTRACTED_DIR / row.relative_path, group=row.group)
        steps.append(out["steps"])
        series.append(out["timeseries"])
        s = out["steps"]
        meta.append({"dataset": "randomized_usage", "cell": row.cell, "group": row.group,
                     "n_operations": len(s),
                     "n_discharge": int((s.comment == "reference discharge").sum()),
                     "procedure": out["procedure"], "description": out["description"]})
        print(f"  {row.cell}: {len(s)} steps ({time.time() - t0:.1f}s)")
    pd.concat(steps, ignore_index=True).to_parquet(INTERIM_DIR / "rw_steps.parquet", index=False)
    pd.concat(series, ignore_index=True).to_parquet(INTERIM_DIR / "rw_timeseries.parquet", index=False)
    return meta


def run() -> None:
    manifest = extract.run()  # no-op if already extracted
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    print("Battery Aging dataset")
    meta = build_battery_aging(manifest)
    print("Randomized Usage dataset")
    meta += build_randomized_usage(manifest)
    pd.DataFrame(meta).to_csv(INTERIM_DIR / "cells.csv", index=False)
    print(f"Wrote tables to {INTERIM_DIR}")


if __name__ == "__main__":
    run()
