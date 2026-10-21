"""State-of-Health estimation experiments.

Question asked: *given the signals available at a cycle, how well can SoH be
estimated without running a full capacity test?*  The measured capacity (and
anything derived from it) is therefore never an input.

Feature sets are nested so their value can be compared:
  Battery Aging      A condition+usage+charge+impedance  (no discharge information)
                     B + early discharge window (<= 600 s)
                     C + partial curve (time to 3.8 V / 3.6 V)
                     LEAKAGE CHECK: C + full discharge (duration/energy to 2.7 V)
  Randomized Usage   A condition+usage (operational data only, no reference test)
                     B + reference charge (CC/CV durations)
                     C + pulse resistance
                     LEAKAGE CHECK: C + reference discharge energy/duration

Validation: leave-one-cell-out, chronological (first 60 % of each cell's life ->
last 40 %), and for Randomized Usage also leave-one-experiment-group-out.

Usage:
    python -m src.models.train_soh
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.evaluation.metrics import summarise_predictions
from src.evaluation.splits import chronological, leave_cells_out, leave_groups_out
from src.features.battery_aging import FEATURE_GROUPS as BA_GROUPS
from src.features.randomized_usage import FEATURE_GROUPS as RW_GROUPS
from src.models.common import cross_validate, feature_importance, feature_ranges, make_models, make_pipeline
from src.visualization.style import CATEGORICAL, INK_2, apply_style, save

RESULTS = REPORTS_DIR / "results"


def _sets(groups: dict, steps: list[list[str]]) -> dict[str, list[str]]:
    out, acc = {}, []
    for name, extra in steps:
        for g in extra:
            acc = acc + [f for f in groups[g] if f not in acc]
        out[name] = list(acc)
    return out


BA_SETS = _sets(BA_GROUPS, [
    ["A: charge+impedance", ["condition", "usage", "charge", "impedance"]],
    ["B: +early discharge", ["early"]],
    ["C: +partial curve", ["partial"]],
    ["LEAK: +full discharge", ["full"]],
])
RW_SETS = _sets(RW_GROUPS, [
    ["A: operational only", ["condition", "usage"]],
    ["B: +reference charge", ["charge"]],
    ["C: +pulse resistance", ["pulse"]],
    ["LEAK: +reference discharge", ["full"]],
])


def load_ba() -> pd.DataFrame:
    ba = pd.read_parquet(PROCESSED_DIR / "ba_cycles.parquet")
    return ba[ba.in_model_cohort & ba.is_valid & ba.soh.notna()].reset_index(drop=True)


def load_rw() -> pd.DataFrame:
    rw = pd.read_parquet(PROCESSED_DIR / "rw_checks.parquet")
    return rw[rw.is_valid & rw.soh.notna()].reset_index(drop=True)


def run_dataset(name: str, df: pd.DataFrame, sets: dict, order_col: str, group_col: str | None):
    models = make_models()
    preds = []
    schemes = [("leave-one-cell-out", "cells", lambda: leave_cells_out(df)),
               ("chronological 60/40", "chronological", lambda: chronological(df, order_col))]
    if group_col:
        schemes.append(("leave-one-group-out", "groups", lambda: leave_groups_out(df, group_col)))
    for set_name, feats in sets.items():
        for scheme, kind, splitter in schemes:
            p = cross_validate(df, feats, "soh", splitter(), kind, models, order_col=order_col,
                               extra_cols=[order_col])
            p["feature_set"], p["validation"], p["dataset"] = set_name, scheme, name
            preds.append(p)
            print(f"  {name} | {set_name:28s} | {scheme:22s} done")
    return pd.concat(preds, ignore_index=True)


def plot_results(summary: pd.DataFrame, preds: pd.DataFrame, dataset: str, tag: str) -> None:
    s = summary[(summary.dataset == dataset) & (summary.validation == "leave-one-cell-out")]
    models = list(dict.fromkeys(s.model))
    sets = list(dict.fromkeys(s.feature_set))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        vals = [s[(s.model == m) & (s.feature_set == fs)].MAE.iloc[0] * 100 for fs in sets]
        ax.bar(np.arange(len(sets)) + k * w, vals, width=w * 0.92, color=CATEGORICAL[k], label=m)
    ax.set_xticks(np.arange(len(sets)) + w * (len(models) - 1) / 2, sets)
    ax.set_ylabel("MAE (SoH percentage points)")
    ax.set_title(f"{dataset}: SoH error on unseen cells (leave-one-cell-out)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(models), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    save(fig, FIGURES_DIR / "models" / f"soh_{tag}_mae_by_feature_set.png")

    # Predicted vs true for the best non-leaky configuration.
    ok = s[~s.feature_set.str.startswith("LEAK") & (s.model != "Mean baseline")]
    best = ok.sort_values("MAE").iloc[0]
    p = preds[(preds.dataset == dataset) & (preds.validation == "leave-one-cell-out")
              & (preds.model == best.model) & (preds.feature_set == best.feature_set)]
    cells = sorted(p.cell.unique(), key=lambda c: int("".join(ch for ch in c if ch.isdigit())))
    show = cells[:: max(1, len(cells) // 6)][:6]
    fig, axes = plt.subplots(1, len(show), figsize=(2.6 * len(show), 2.6), sharey=True)
    # preds holds both datasets, so the column always exists; RW rows have it all-NaN.
    xcol = "cycle_number" if p.get("cycle_number", pd.Series(dtype=float)).notna().any() else "efc"
    for ax, cell in zip(np.atleast_1d(axes), show):
        d = p[p.cell == cell].sort_values(xcol)
        ax.plot(d[xcol], d.y_true, color=CATEGORICAL[0], lw=1.6, label="measured")
        ax.plot(d[xcol], d.y_pred, color=CATEGORICAL[1], lw=1.2, label="predicted")
        ax.set_title(cell, fontsize=9)
        ax.set_xlabel("cycle" if xcol == "cycle_number" else "EFC")
    np.atleast_1d(axes)[0].set_ylabel("SoH")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle(f"{dataset}: held-out cells, {best.model}, {best.feature_set}",
                 x=0.01, ha="left", color=INK_2, fontsize=9)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "models" / f"soh_{tag}_heldout_cells.png")


def fit_final(df, feats, model_name, path, meta):
    model = make_models()[model_name]
    pipe = make_pipeline(model).fit(df[feats], df.soh)
    joblib.dump(pipe, path)
    path.with_suffix(".json").write_text(json.dumps(
        meta | {"features": feats, "feature_ranges": feature_ranges(df, feats)}, indent=2))


def run() -> None:
    apply_style()
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ba, rw = load_ba(), load_rw()
    print(f"Battery Aging: {len(ba)} cycles from {ba.cell.nunique()} cells; "
          f"Randomized Usage: {len(rw)} checks from {rw.cell.nunique()} cells")
    preds = pd.concat([
        run_dataset("Battery Aging", ba, BA_SETS, "cycle_number", None),
        run_dataset("Randomized Usage", rw, RW_SETS, "efc", "group"),
    ], ignore_index=True)
    preds.to_parquet(RESULTS / "soh_predictions.parquet", index=False)
    summary = summarise_predictions(preds, ["dataset", "validation", "feature_set", "model"])
    summary.to_csv(RESULTS / "soh_metrics.csv", index=False)

    per_cell = summarise_predictions(
        preds[preds.validation == "leave-one-cell-out"],
        ["dataset", "feature_set", "model", "cell"])
    per_cell.to_csv(RESULTS / "soh_metrics_per_cell.csv", index=False)

    plot_results(summary, preds, "Battery Aging", "ba")
    plot_results(summary, preds, "Randomized Usage", "rw")

    # Final models: best non-leaky configuration under leave-one-cell-out, refit on all cells.
    for dataset, df, sets, tag in (("Battery Aging", ba, BA_SETS, "ba"),
                                   ("Randomized Usage", rw, RW_SETS, "rw")):
        s = summary[(summary.dataset == dataset) & (summary.validation == "leave-one-cell-out")
                    & ~summary.feature_set.str.startswith("LEAK") & (summary.model != "Mean baseline")]
        best = s.sort_values("MAE").iloc[0]
        fit_final(df, sets[best.feature_set], best.model, MODELS_DIR / f"soh_{tag}.joblib",
                  {"dataset": dataset, "model": best.model, "feature_set": best.feature_set,
                   "loco_MAE": best.MAE, "loco_RMSE": best.RMSE, "loco_R2": best.R2,
                   "target": "soh"})
        imp = feature_importance(df, sets[best.feature_set], "soh")
        imp.to_csv(RESULTS / f"soh_{tag}_feature_importance.csv", header=["importance"])

    pd.set_option("display.width", 200)
    cols = ["dataset", "validation", "feature_set", "model", "MAE", "RMSE", "R2", "MAE_fold_std"]
    print(summary[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    run()
