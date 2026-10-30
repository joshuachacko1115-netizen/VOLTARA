"""Inference for the EV pack models (models/ev_*.joblib from src.models.train_ev).

Input: raw rows in the original CSV format of either EV dataset, for one or more
batteries / vehicles, oldest first. Targets may be absent. The rows go through the
same cleaning (src/preprocessing/ev_cleaning.py) and causal feature code
(src/features/ev_features.py) as in training, so a battery's earlier rows are
needed for the history features: pass its whole history, not only the last row.

Leakage columns are ignored even if present (`capacity_fade_pct`,
`Range_Remaining_km`): they are renamed `leak_*` by the cleaning step and no model
uses them.

Outputs per row: one column per model, `pred_<task>` (a value, or the most likely
class with `p_<task>_<class>` probabilities), and `inputs_outside_training_range`.

Usage:
    python -m src.inference.ev_predict --dataset ev_degradation --unit BATT_00001
    python -m src.inference.ev_predict --dataset ev_telematics --csv my_trips.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.data.ev_loaders import EV1_DATE_COLS, find_ev_files, snake
from src.features.ev_features import ev1_features, ev2_features
from src.preprocessing.ev_cleaning import clean_ev_degradation, clean_ev_telematics

UNITS = {"ev_degradation": "battery_id", "ev_telematics": "vehicle_id"}


class EVPredictor:
    def __init__(self, dataset: str, models_dir: Path = MODELS_DIR):
        if dataset not in UNITS:
            raise ValueError(f"dataset must be one of {list(UNITS)}")
        self.dataset = dataset
        self.models = {}
        for path in sorted(models_dir.glob("ev_*.joblib")):
            obj = joblib.load(path)
            if obj["meta"]["dataset"] == dataset:
                self.models[obj["meta"]["task"]] = obj
        if not self.models:
            raise FileNotFoundError(f"no ev_* models for {dataset} in {models_dir}; run python -m src.models.train_ev")

    def prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        if self.dataset == "ev_degradation":
            for c in EV1_DATE_COLS:
                if c in raw:
                    raw[c] = pd.to_datetime(raw[c])
            clean, _ = clean_ev_degradation(raw, require_targets=False)
            return ev1_features(clean)[0]
        raw.columns = [snake(c) for c in raw.columns]
        raw["trip_date"] = pd.to_datetime(raw.trip_date)
        clean, _ = clean_ev_telematics(raw)
        return ev2_features(clean)[0]

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare(raw)
        unit = UNITS[self.dataset]
        id_col = "observation_id" if self.dataset == "ev_degradation" else "trip_id"
        out = df[[unit, id_col]].copy()
        outside: list[list[str]] = [[] for _ in range(len(df))]
        for task, obj in self.models.items():
            meta, pipe = obj["meta"], obj["pipeline"]
            X = df.reindex(columns=meta["categorical_features"] + meta["numeric_features"])
            if meta["kind"] == "regression":
                out[f"pred_{task}"] = pipe.predict(X)
            else:
                proba = pipe.predict_proba(X)
                out[f"pred_{task}"] = [meta["classes"][i] for i in proba.argmax(1)]
                for i, c in enumerate(meta["classes"]):
                    out[f"p_{task}_{c}"] = proba[:, i]
            for f, (lo, hi) in meta["feature_ranges"].items():
                if lo is None or f not in df:
                    continue
                bad = np.flatnonzero(~df[f].between(lo, hi) & df[f].notna())
                for i in bad:
                    if f not in outside[i]:
                        outside[i].append(f)
        out["inputs_outside_training_range"] = [", ".join(x) for x in outside]
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ev_degradation", choices=list(UNITS))
    ap.add_argument("--unit", help="score one battery / vehicle from the raw dataset (demo)")
    ap.add_argument("--csv", help="score your own CSV in the dataset's original format")
    ap.add_argument("--rows", type=int, default=5)
    args = ap.parse_args()
    if args.csv:
        raw = pd.read_csv(args.csv)
    else:   # demo: one unit from the raw file, in its original column format
        raw = pd.read_csv(find_ev_files()[args.dataset])
        unit = "battery_id" if args.dataset == "ev_degradation" else "Vehicle_ID"
        raw = raw[raw[unit] == (args.unit or raw[unit].iloc[0])]
    pred = EVPredictor(args.dataset).predict(raw)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(pred.tail(args.rows).to_string(index=False))


if __name__ == "__main__":
    main()
