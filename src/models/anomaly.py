"""Unsupervised anomaly detection with Isolation Forest.

What counts as "unusual" depends on the cell (a 4 degC cell at 1 A looks nothing
like a 24 degC cell at 2 A), so every feature is first expressed as a robust
z-score relative to *its own cell*: (x - cell median) / (1.4826 * cell MAD).
Isolation Forest then looks for cycles whose combination of deviations is rare.

This is a retrospective analysis of complete test records - the cell medians use
each cell's whole history - so it is used for data understanding and is never fed
into the supervised models.

An anomaly is a statistically unusual measurement, NOT a diagnosed fault. Each
flagged row is explained by its largest deviations plus context that often has a
benign cause (long rest before the cycle -> capacity regeneration; change of
ambient temperature or load current -> different test condition; rule-based data
quality flags -> likely a recording / test-rig issue).

Usage:
    python -m src.models.anomaly
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR
from src.features.targets import BA_FLAG_COLUMNS
from src.preprocessing.cleaning import MAD_TO_STD
from src.visualization.style import CATEGORICAL, INK_2, MUTED, STATUS, apply_style, save

RESULTS = REPORTS_DIR / "results"

BA_FEATURES = ["capacity_ah_2v7", "cap_resid_prev5", "v_at_60s", "v_at_300s", "dis_t_start",
               "t_rise_300s", "dis_t_max", "dis_v_std", "chg_cc_duration_s", "chg_cv_duration_s",
               "chg_t_max", "i_set"]
RW_FEATURES = ["soh", "cap_resid_prev3", "ref_chg_cc_duration_s", "ref_chg_cv_duration_s",
               "r_operational_ohm", "r_pulse_ohm", "interval_rw_t_mean", "interval_rw_t_max",
               "interval_rw_i_mean", "ref_t_max"]


def robust_z_within_cell(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    z = pd.DataFrame(index=df.index)
    for f in features:
        g = df.groupby("cell")[f]
        med = g.transform("median")
        mad = g.transform(lambda s: (s - s.median()).abs().median())
        scale = (MAD_TO_STD * mad).where(lambda s: s > 1e-9)
        # Fallback for near-constant features (e.g. i_set): use the cell's std.
        scale = scale.fillna(g.transform("std")).replace(0, np.nan)
        z[f] = ((df[f] - med) / scale).clip(-20, 20)
    return z


def detect(df: pd.DataFrame, features: list[str], save_tag: str | None = None) -> pd.DataFrame:
    z = robust_z_within_cell(df, features)
    X = z.fillna(0.0)  # missing -> "typical for this cell"
    forest = IsolationForest(n_estimators=500, contamination="auto", random_state=RANDOM_SEED)
    forest.fit(X)
    if save_tag:  # for src/inference: the forest scores within-battery z-scores of these features
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(forest, MODELS_DIR / f"anomaly_{save_tag}.joblib")
        (MODELS_DIR / f"anomaly_{save_tag}.json").write_text(json.dumps(
            {"features": features, "input": "robust z-score of each feature within one battery's history",
             "n_training_rows": len(X), "n_estimators": 500, "contamination": "auto"}, indent=2))
    out = df.copy()
    out["anomaly_score"] = -forest.score_samples(X)        # higher = more unusual
    out["is_anomaly"] = forest.predict(X) == -1
    # Explanation: the three features furthest from the cell's typical value.
    def fmt(v):  # z was clipped at +-20
        return f"{'≥' if v >= 20 else '≤' if v <= -20 else ''}{v:+.1f}σ"
    out["top_deviations"] = [
        "; ".join(f"{f} {fmt(z.loc[i, f])}" for f in z.loc[i].abs().nlargest(3).index)
        for i in z.index]
    return out


def ba_context(row) -> str:
    notes = []
    flags = [f.replace("flag_", "") for f in BA_FLAG_COLUMNS if bool(row.get(f, False))]
    if flags:
        notes.append("fails data-quality rules: " + ", ".join(flags))
    if row.get("rest_before_h", 0) and row.rest_before_h > 24:
        notes.append(f"follows a {row.rest_before_h:.0f} h rest (capacity regeneration is common)")
    if row.get("rare_condition", False):
        notes.append(f"run at a different test condition from this cell's main one "
                     f"({row.ambient_temperature:.0f} °C, {row.i_set:.1f} A)")
    elif row.get("ambient_changed", False):
        notes.append("ambient temperature differs from the previous cycle")
    elif row.get("current_changed", False):
        notes.append("load current differs from the previous cycle")
    if not notes:
        notes.append("no benign explanation found in the recorded context - worth inspecting")
    return " | ".join(notes)


def run_ba() -> pd.DataFrame:
    ba = pd.read_parquet(PROCESSED_DIR / "ba_cycles.parquet")
    ba = ba[ba.capacity_ah_2v7.notna() | ba.i_set.notna()].copy()
    ba["ambient_changed"] = ba.groupby("cell").ambient_temperature.diff().abs().fillna(0) > 0
    ba["current_changed"] = ba.groupby("cell").i_set.diff().abs().fillna(0) > 0.3
    cond = ba.ambient_temperature.round().astype(str) + "_" + ba.i_set.round().astype(str)
    main = cond.groupby(ba.cell).transform(lambda s: s.value_counts().index[0])
    ba["rare_condition"] = cond != main  # not the cell's main (most common) test condition
    res = detect(ba, BA_FEATURES, save_tag="ba")
    res["context"] = res.apply(ba_context, axis=1)
    res["has_quality_flag"] = res[BA_FLAG_COLUMNS].any(axis=1)
    return res


def run_rw() -> pd.DataFrame:
    rw = pd.read_parquet(PROCESSED_DIR / "rw_checks.parquet")
    rw = rw[rw.is_valid].copy()
    res = detect(rw, RW_FEATURES, save_tag="rw")
    res["context"] = np.where(res.interval_n_bad_temperature > 0,
                              "temperature sensor drop-out in the preceding interval", "")
    res["context"] = res.context.where(res.context != "",
                                       np.where(res.capacity_spread_ah > 0.05,
                                                "the two reference discharges of this check disagree "
                                                "by > 0.05 Ah", ""))
    res["context"] = res.context.where(
        res.context != "", np.where(res.check_id <= 3, "early-life check (cells often settle in the "
                                    "first checks)", "no recorded context - worth inspecting"))
    return res


def plot_ba(res: pd.DataFrame) -> None:
    cells = ["B0005", "B0006", "B0018", "B0033", "B0036", "B0047"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.4))
    for ax, cell in zip(axes.flat, cells):
        d = res[res.cell == cell]
        ax.plot(d.cycle_number, d.capacity_ah_2v7, color=CATEGORICAL[0], lw=1.2)
        a = d[d.is_anomaly]
        ax.scatter(a.cycle_number, a.capacity_ah_2v7, s=40, facecolor="none",
                   edgecolor=STATUS["critical"], lw=1.4, zorder=3, label="Isolation Forest anomaly")
        q = d[d.has_quality_flag]
        ax.scatter(q.cycle_number, q.capacity_ah_2v7, marker="x", s=16, color=MUTED, lw=1,
                   zorder=4, label="fails quality rules")
        ax.set_title(f"{cell}: {int(d.is_anomaly.sum())} anomalies", fontsize=9)
        ax.set_xlabel("Discharge cycle")
        ax.set_ylabel("Capacity (Ah)")
    axes.flat[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Battery Aging: anomalous cycles (circled) on the capacity trajectory",
                 x=0.01, ha="left", color=INK_2)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "anomaly" / "ba_anomalies_on_capacity.png")

    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.hist(res.anomaly_score[~res.is_anomaly], bins=50, color=CATEGORICAL[0], label="normal")
    ax.hist(res.anomaly_score[res.is_anomaly], bins=50, color=STATUS["critical"], label="anomaly")
    ax.set_xlabel("Isolation Forest anomaly score (higher = more unusual)")
    ax.set_ylabel("cycles")
    ax.set_title("Battery Aging: anomaly score distribution")
    ax.legend()
    save(fig, FIGURES_DIR / "anomaly" / "ba_anomaly_scores.png")


def plot_rw(res: pd.DataFrame) -> None:
    cells = res.groupby("cell").is_anomaly.sum().sort_values(ascending=False).index[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.4))
    for ax, cell in zip(axes.flat, cells):
        d = res[res.cell == cell]
        ax.plot(d.efc, d.soh, color=CATEGORICAL[0], lw=1.2, marker="o", ms=3)
        a = d[d.is_anomaly]
        ax.scatter(a.efc, a.soh, s=60, facecolor="none", edgecolor=STATUS["critical"], lw=1.4, zorder=3)
        ax.set_title(f"{cell}: {int(d.is_anomaly.sum())} anomalous checks", fontsize=9)
        ax.set_xlabel("EFC")
        ax.set_ylabel("SoH")
    fig.suptitle("Randomized Usage: anomalous reference checks (circled), cells with most anomalies",
                 x=0.01, ha="left", color=INK_2)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "anomaly" / "rw_anomalies_on_soh.png")


def run() -> None:
    apply_style()
    RESULTS.mkdir(parents=True, exist_ok=True)
    ba = run_ba()
    rw = run_rw()
    cols_ba = ["cell", "cycle_number", "capacity_ah_2v7", "anomaly_score", "top_deviations", "context"]
    ba[ba.is_anomaly].sort_values("anomaly_score", ascending=False)[cols_ba].to_csv(
        RESULTS / "anomalies_ba.csv", index=False)
    cols_rw = ["cell", "check_id", "efc", "soh", "anomaly_score", "top_deviations", "context"]
    rw[rw.is_anomaly].sort_values("anomaly_score", ascending=False)[cols_rw].to_csv(
        RESULTS / "anomalies_rw.csv", index=False)
    plot_ba(ba)
    plot_rw(rw)

    # How do anomalies relate to the rule-based quality flags?
    ct = pd.crosstab(ba.is_anomaly, ba.has_quality_flag, margins=True)
    ct.to_csv(RESULTS / "anomalies_ba_vs_quality_flags.csv")
    print(f"Battery Aging: {ba.is_anomaly.sum()} of {len(ba)} cycles flagged")
    print("Isolation Forest anomaly (rows) vs rule-based quality flag (columns):\n", ct)
    benign = ba[ba.is_anomaly].context.str.contains("rest|differ|quality").mean()
    print(f"Share of BA anomalies with a recorded benign/data explanation: {benign:.0%}")
    print(f"Randomized Usage: {rw.is_anomaly.sum()} of {len(rw)} checks flagged")
    print(ba[ba.is_anomaly].sort_values("anomaly_score", ascending=False)[cols_ba].head(8).to_string(index=False))
    print(rw[rw.is_anomaly].sort_values("anomaly_score", ascending=False)[cols_rw].head(6).to_string(index=False))


if __name__ == "__main__":
    run()
