"""90 % prediction intervals for the EV regression models (split-conformal, by unit).

For each saved model (models/ev_<task>.joblib) the out-of-fold residuals on the dev
batteries / vehicles (GroupKFold, the same folds and settings as training) set the
interval half-width: the conformal (1 - alpha) quantile of |y - prediction|. Errors grow
with the predicted value (a young battery's RUL is less certain than an old one's), so the
quantile is taken separately in 4 bins of the predicted value (Mondrian conformal).

The intervals are then checked once on the held-out test units: coverage should be close
to 90 %. The result is stored in each model's meta as `interval` (and, for the manual-entry
models, `interval_categories_unknown`, calibrated on rows with chemistry/climate blanked).

    python -m src.models.ev_intervals
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from src.config import EV_DIR, EV_EOL_SOH_PCT, MODELS_DIR, REPORTS_DIR
from src.models.common import apply_interval
from src.models.train_ev import QUICK_CATEGORICAL, Task, oof_predict, with_unknown_categories

LEVEL = 0.90
N_BINS = 4
TASKS = ["degradation_soh", "degradation_rul", "telematics_soh", "quick_soh", "quick_rul"]
BOUNDS = {"soh": (0.0, 100.0), "rul": (0.0, None)}


def conformal_quantile(abs_resid: np.ndarray, level: float = LEVEL) -> float:
    n = len(abs_resid)
    k = min(int(np.ceil((n + 1) * level)), n)
    return float(np.sort(abs_resid)[k - 1])


def fit_interval(pred: np.ndarray, y: np.ndarray) -> dict:
    edges = np.quantile(pred, np.linspace(0, 1, N_BINS + 1))[1:-1]
    bins = np.digitize(pred, edges)
    half = [conformal_quantile(np.abs(y - pred)[bins == b]) for b in range(N_BINS)]
    return {"level": LEVEL, "method": f"split conformal, {N_BINS} bins of the predicted value",
            "bin_edges": [float(e) for e in edges], "half_width": half}


def check(interval: dict, pred: np.ndarray, y: np.ndarray, bounds) -> dict:
    lo, hi = apply_interval(interval, pred, bounds)
    inside = (y >= lo) & (y <= hi)
    bins = np.digitize(pred, interval["bin_edges"])
    return {"test_coverage": float(inside.mean()), "test_mean_width": float(np.mean(hi - lo)),
            "test_coverage_by_bin": [float(inside[bins == b].mean()) if (bins == b).any() else None
                                     for b in range(N_BINS)]}


def task_from_meta(meta: dict) -> Task:
    return Task(meta["task"], meta["dataset"], meta["target"], meta["kind"], meta["unit"],
                meta["numeric_features"], meta["categorical_features"])


def calibrate(key: str, frames: dict) -> list[dict]:
    t0 = time.time()
    path = MODELS_DIR / f"ev_{key}.joblib"
    obj = joblib.load(path)
    meta, pipe = obj["meta"], obj["pipeline"]
    task = task_from_meta(meta)
    family = "XGBoost" if meta["model"].startswith("XGBoost") else meta["model"]
    df = frames[meta["dataset"]]
    bounds = BOUNDS["rul" if "rul" in key else "soh"]
    quick = key.startswith("quick")
    data = with_unknown_categories(df) if quick else df
    dev = data[data.split == "dev"].reset_index(drop=True)
    oof = oof_predict(task, dev, family, meta["xgb_params"])
    y_dev = dev[task.target].to_numpy(float)
    test = df[df.split == "test"].reset_index(drop=True)
    y_te = test[task.target].to_numpy(float)

    rows = []
    variants = [("interval", slice(None) if not quick else slice(0, len(dev) // 2), test)]
    if quick:          # second half of dev = the copy with categories blanked
        blank = test.copy()
        blank[QUICK_CATEGORICAL] = np.nan
        variants.append(("interval_categories_unknown", slice(len(dev) // 2, None), blank))
    for name, rows_dev, te in variants:
        iv = fit_interval(oof[rows_dev], y_dev[rows_dev])
        p_te = pipe.predict(te[task.cats + task.nums])
        iv.update(check(iv, p_te, y_te, bounds))
        meta[name] = iv
        rows.append({"task": key, "variant": name, "coverage": iv["test_coverage"],
                     "mean_width": iv["test_mean_width"], "half_width_by_bin": iv["half_width"],
                     "coverage_by_bin": iv["test_coverage_by_bin"]})
        print(f"  {key:16s} {name:28s} test coverage {iv['test_coverage']:.3f}  mean width "
              f"{iv['test_mean_width']:.3g}  half-widths {[round(h, 3) for h in iv['half_width']]}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    joblib.dump({"pipeline": pipe, "meta": meta}, path)
    (MODELS_DIR / f"ev_{key}.json").write_text(json.dumps(meta, indent=2, default=str))
    return rows


def run() -> pd.DataFrame:
    frames = {d: pd.read_parquet(EV_DIR / f"{d}_features.parquet") for d in ("ev_degradation", "ev_telematics")}
    print(f"[intervals] {LEVEL:.0%} split-conformal intervals; EoL at {EV_EOL_SOH_PCT:g} % SoH", flush=True)
    rows = [r for key in TASKS for r in calibrate(key, frames)]
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "results" / "ev_interval_coverage.csv", index=False)
    print("[intervals] done -> reports/results/ev_interval_coverage.csv", flush=True)
    return out


if __name__ == "__main__":
    run()
