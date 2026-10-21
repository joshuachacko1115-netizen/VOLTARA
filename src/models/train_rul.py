"""Remaining-Useful-Life prediction experiments.

Target: Battery Aging -> cycles until EOL; Randomized Usage -> EFC until EOL
(EOL = smoothed SoH <= 80 %, see src/features/targets.py). Only cells that reach
EOL have labels; censored cells (B0025-B0029, B0031, B0032, RW26-RW28) are reported
separately with their known lower bound and are never scored as if they had failed.

Inputs are things known at the current cycle: the measured SoH so far (causal
rolling median), its recent trend, usage, temperature, resistance. Future
capacity is never used as a feature.

Models
  Mean baseline           predicts the training-set mean RUL
  Linear extrapolation    no training: extends the recent SoH trend to 80 %
  Linear Regression, Random Forest, Gradient Boosting, (XGBoost)
  CNN-LSTM                over the last 10 cycles (BA) / 4 checks (RW), if PyTorch is available

Validation: leave-one-cell-out (and leave-one-group-out for Randomized Usage).

Usage:
    python -m src.models.train_rul [--no-deep]
"""
from __future__ import annotations

import argparse
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (BA_RATED_CAPACITY_AH, EOL_SOH, FIGURES_DIR, MODELS_DIR, PROCESSED_DIR,
                        REPORTS_DIR)
from src.evaluation.metrics import summarise_predictions
from src.evaluation.splits import check_split, leave_cells_out, leave_groups_out
from src.models.common import cross_validate, feature_importance, feature_ranges, make_models, make_pipeline
from src.models.sequence_model import HAVE_TORCH, CNNLSTMRegressor
from src.visualization.style import CATEGORICAL, INK_2, MUTED, apply_style, save

RESULTS = REPORTS_DIR / "results"

BA_FEATURES = ["soh_roll_med5", "cap_roll_std5", "fade_slope_10", "fade_slope_20",
               "capacity_fade_ah", "cycle_number", "i_set", "ambient_temperature", "re_ohm",
               "rct_ohm", "chg_cc_duration_s", "dis_t_max", "rest_before_h"]
RW_FEATURES = ["soh_roll_med3", "fade_slope_4", "fade_slope_8", "efc", "elapsed_days",
               "interval_rw_i_mean", "interval_rw_t_mean", "interval_frac_hit_vmin",
               "r_operational_ohm", "r_pulse_ohm", "ref_chg_cc_duration_s",
               "ambient_40c", "dist_skewed_high", "dist_skewed_low", "dist_uniform"]

DATASETS = {
    "Battery Aging": dict(file="ba_cycles.parquet", target="rul_cycles", order="cycle_number",
                          features=BA_FEATURES, soh="soh_roll_med5", slope="fade_slope_20",
                          slope_to_soh_per_unit=1 / 100 / BA_RATED_CAPACITY_AH, window=10,
                          group_col=None, tag="ba", unit="cycles"),
    # Same protocol only (2 A CC, 24 degC, identical charge): isolates cell-to-cell
    # variation from changes in test conditions.
    "Battery Aging (B0005/06/07/18)": dict(file="ba_cycles.parquet", target="rul_cycles",
                          order="cycle_number", features=BA_FEATURES, soh="soh_roll_med5",
                          slope="fade_slope_20", slope_to_soh_per_unit=1 / 100 / BA_RATED_CAPACITY_AH,
                          window=10, group_col=None, tag="ba_same_protocol", unit="cycles",
                          cells=["B0005", "B0006", "B0007", "B0018"], final_model=False),
    "Randomized Usage": dict(file="rw_checks.parquet", target="rul_efc", order="efc",
                             features=RW_FEATURES, soh="soh_roll_med3", slope="fade_slope_8",
                             slope_to_soh_per_unit=1 / 100, window=4, group_col="group",
                             tag="rw", unit="EFC"),
}


def load(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(PROCESSED_DIR / cfg["file"])
    if cfg["tag"].startswith("ba"):
        labelled = df[df.in_rul_cohort & df.is_valid & df.rul_cycles.notna()]
        censored = df[df.in_model_cohort & df.censored & df.is_valid]
    else:
        labelled = df[df.is_valid & df.rul_efc.notna()]
        censored = df[df.censored & df.is_valid]
    return labelled.reset_index(drop=True), censored.reset_index(drop=True)


def extrapolation_baseline(df, cfg, cap: float) -> np.ndarray:
    """RUL = (SoH_now - EOL) / fade rate, using only the trailing trend at each row."""
    soh = df[cfg["soh"]].to_numpy()
    rate = -df[cfg["slope"]].to_numpy() * cfg["slope_to_soh_per_unit"]  # SoH lost per unit
    with np.errstate(divide="ignore", invalid="ignore"):
        rul = np.where(rate > 0, (soh - EOL_SOH) / rate, cap)
    rul = np.where(soh <= EOL_SOH, 0, rul)
    return np.clip(np.nan_to_num(rul, nan=cap), 0, cap)


def run_dataset(name: str, cfg: dict, deep: bool) -> pd.DataFrame:
    df, censored = load(cfg)
    if cfg.get("cells"):
        df = df[df.cell.isin(cfg["cells"])].reset_index(drop=True)
    print(f"{name}: {len(df)} labelled rows from {df.cell.nunique()} cells "
          f"({censored.cell.nunique()} censored cells kept aside)")
    schemes = [("leave-one-cell-out", "cells", lambda: leave_cells_out(df))]
    if cfg["group_col"]:
        schemes.append(("leave-one-group-out", "groups", lambda: leave_groups_out(df, cfg["group_col"])))
    preds = []
    for scheme, kind, splitter in schemes:
        p = cross_validate(df, cfg["features"], cfg["target"], splitter(), kind, make_models(),
                           extra_cols=[cfg["order"], cfg["soh"]])
        extra = []
        for fold, train, test in splitter():
            check_split(df, train, test, kind)
            te = df.loc[test]
            cap = float(df.loc[train, cfg["target"]].max())
            base = te[["cell", cfg["order"], cfg["soh"]]].copy()
            base["fold"], base["y_true"] = fold, te[cfg["target"]].to_numpy()
            extra.append(base.assign(model="Linear extrapolation",
                                     y_pred=extrapolation_baseline(te, cfg, cap)))
            if deep and HAVE_TORCH:
                m = CNNLSTMRegressor(cfg["features"], cfg["order"], window=cfg["window"],
                                     n_val_cells=1 if cfg["tag"] == "ba" else 3)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(df.loc[train], df.loc[train, cfg["target"]])
                extra.append(base.assign(model="CNN-LSTM", y_pred=m.predict(te)))
        p = pd.concat([p, *extra], ignore_index=True)
        p["validation"], p["dataset"] = scheme, name
        preds.append(p)
        print(f"  {scheme} done")
    return pd.concat(preds, ignore_index=True)


def plots(preds: pd.DataFrame, summary: pd.DataFrame, name: str, cfg: dict) -> None:
    tag, unit = cfg["tag"], cfg["unit"]
    p = preds[(preds.dataset == name) & (preds.validation == "leave-one-cell-out")]
    s = summary[(summary.dataset == name) & (summary.validation == "leave-one-cell-out")]
    order = s.sort_values("MAE").model.tolist()

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(order, s.set_index("model").loc[order, "MAE"], color=CATEGORICAL[0], height=0.6)
    ax.invert_yaxis()
    ax.set_xlabel(f"MAE ({unit})")
    ax.set_title(f"{name}: RUL error on unseen cells (leave-one-cell-out)")
    ax.grid(axis="y", visible=False)
    for y, m in enumerate(order):
        ax.text(s.set_index("model").loc[m, "MAE"], y, f"  {s.set_index('model').loc[m, 'MAE']:.1f}",
                va="center", fontsize=8, color=INK_2)
    save(fig, FIGURES_DIR / "models" / f"rul_{tag}_model_comparison.png")

    # Error vs position in life: is the model worse early or late?
    best = next(m for m in order if m not in ("Mean baseline",))
    b = p[p.model == best].copy()
    b["abs_err"] = (b.y_pred - b.y_true).abs()
    b["rul_bin"] = pd.cut(b.y_true, bins=[-0.1, 10, 25, 50, 100, 200, 400, 1e9] if tag == "ba"
                          else [-0.1, 25, 50, 100, 150, 200, 300, 1e9])
    e = b.groupby("rul_bin", observed=True).abs_err.agg(["median", "count"])
    e.to_csv(RESULTS / f"rul_{tag}_error_by_true_rul.csv")

    cells = sorted(p.cell.unique(), key=lambda c: int("".join(ch for ch in c if ch.isdigit())))
    show = cells if len(cells) <= 6 else cells[:: max(1, len(cells) // 6)][:6]
    fig, axes = plt.subplots(1, len(show), figsize=(2.7 * len(show), 2.7))
    for ax, cell in zip(np.atleast_1d(axes), show):
        d = p[(p.cell == cell)]
        t = d[d.model == best].sort_values(cfg["order"])
        ax.plot(t[cfg["order"]], t.y_true, color=CATEGORICAL[0], lw=1.8, label="true RUL")
        ax.plot(t[cfg["order"]], t.y_pred, color=CATEGORICAL[1], lw=1.2, label=best)
        x = d[d.model == "Linear extrapolation"].sort_values(cfg["order"])
        ax.plot(x[cfg["order"]], x.y_pred, color=MUTED, lw=1, ls="--", label="linear extrapolation")
        ax.set_title(cell, fontsize=9)
        ax.set_xlabel("cycle" if tag == "ba" else "EFC")
    np.atleast_1d(axes)[0].set_ylabel(f"RUL ({unit})")
    np.atleast_1d(axes)[0].legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "models" / f"rul_{tag}_heldout_cells.png")


def censored_check(name: str, cfg: dict) -> pd.DataFrame:
    """For censored cells: does the final model at least respect the known lower bound?"""
    df, censored = load(cfg)
    if censored.empty:
        return pd.DataFrame()
    pipe = joblib.load(MODELS_DIR / f"rul_{cfg['tag']}.joblib")
    meta = json.loads((MODELS_DIR / f"rul_{cfg['tag']}.json").read_text())
    if meta["model"] in ("Linear extrapolation", "CNN-LSTM"):
        return pd.DataFrame()
    c = censored.copy()
    c["rul_pred"] = pipe.predict(c[cfg["features"]])
    c["respects_lower_bound"] = c.rul_pred >= c.rul_lower_bound
    return c.groupby("cell").agg(rows=("rul_pred", "size"),
                                 frac_pred_above_lower_bound=("respects_lower_bound", "mean")
                                 ).assign(dataset=name)


def run(deep: bool = True) -> None:
    apply_style()
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_preds = []
    for name, cfg in DATASETS.items():
        all_preds.append(run_dataset(name, cfg, deep))
    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_parquet(RESULTS / "rul_predictions.parquet", index=False)
    summary = summarise_predictions(preds, ["dataset", "validation", "model"])
    summary.to_csv(RESULTS / "rul_metrics.csv", index=False)
    summarise_predictions(preds[preds.validation == "leave-one-cell-out"],
                          ["dataset", "model", "cell"]).to_csv(RESULTS / "rul_metrics_per_cell.csv",
                                                               index=False)
    cens = []
    for name, cfg in DATASETS.items():
        plots(preds, summary, name, cfg)
        if cfg.get("final_model") is False:
            continue
        s = summary[(summary.dataset == name) & (summary.validation == "leave-one-cell-out")
                    & ~summary.model.isin(["Mean baseline", "Linear extrapolation", "CNN-LSTM"])]
        best = s.sort_values("MAE").iloc[0]
        df, _ = load(cfg)
        pipe = make_pipeline(make_models()[best.model]).fit(df[cfg["features"]], df[cfg["target"]])
        joblib.dump(pipe, MODELS_DIR / f"rul_{cfg['tag']}.joblib")
        (MODELS_DIR / f"rul_{cfg['tag']}.json").write_text(json.dumps({
            "dataset": name, "model": best.model, "target": cfg["target"], "unit": cfg["unit"],
            "features": cfg["features"], "loco_MAE": best.MAE, "loco_RMSE": best.RMSE,
            "loco_R2": best.R2, "feature_ranges": feature_ranges(df, cfg["features"])}, indent=2))
        feature_importance(df, cfg["features"], cfg["target"]).to_csv(
            RESULTS / f"rul_{cfg['tag']}_feature_importance.csv", header=["importance"])
        cens.append(censored_check(name, cfg))
    pd.concat(cens).to_csv(RESULTS / "rul_censored_cells_check.csv")
    pd.set_option("display.width", 200)
    print(summary[["dataset", "validation", "model", "n", "MAE", "RMSE", "R2", "MAE_fold_std"]]
          .round(3).to_string(index=False))
    print(pd.concat(cens).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-deep", action="store_true", help="skip the CNN-LSTM")
    run(deep=not ap.parse_args().no_deep)
