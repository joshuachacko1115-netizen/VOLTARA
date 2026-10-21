"""Exploratory data analysis figures for both datasets.

Usage:
    python -m src.visualization.eda

Writes PNGs to reports/figures/eda/ and summary tables to reports/eda/.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (BA_MODEL_CELLS, BA_RATED_CAPACITY_AH, EOL_SOH, FIGURES_DIR, INTERIM_DIR,
                        PROCESSED_DIR, REPORTS_DIR)
from src.features.battery_aging import FEATURE_GROUPS as BA_GROUPS
from src.features.randomized_usage import FEATURE_GROUPS as RW_GROUPS
from src.visualization.style import (CATEGORICAL, DIV_CMAP, INK_2, MUTED, SEQ_CMAP, apply_style,
                                     save)

OUT = FIGURES_DIR / "eda"
TABLES = REPORTS_DIR / "eda"

RW_GROUP_LABELS = {
    "Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post": "Uniform RW charge+discharge, RT",
    "Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post": "Uniform RW discharge, RT",
    "Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post": "Uniform RW discharge, variable charge, RT",
    "RW_Skewed_High_Room_Temp_DataSet_2Post": "Skewed-high current, RT",
    "RW_Skewed_Low_Room_Temp_DataSet_2Post": "Skewed-low current, RT",
    "RW_Skewed_High_40C_DataSet_2Post": "Skewed-high current, 40 °C",
    "RW_Skewed_Low_40C_DataSet_2Post": "Skewed-low current, 40 °C",
}
BA_GROUP_LABELS = {
    "B0005+B0006+B0007+B0018": "2 A CC, 24 °C",
    "B0025+B0026+B0027+B0028": "4 A square wave, 24 °C",
    "B0029+B0030+B0031+B0032": "4 A CC, 43 °C",
    "B0033+B0034": "4 A CC, 24 °C",
    "B0036": "2 A CC, 24 °C",
    "B0038+B0039+B0040": "1/2/4 A, 24 & 44 °C",
    "B0041+B0042+B0043+B0044": "4 A & 1 A, 4 °C",
    "B0045+B0046+B0047+B0048": "1 A CC, 4 °C",
    "B0049+B0050+B0051+B0052": "2 A CC, 4 °C",
    "B0053+B0054+B0055+B0056": "2 A CC, 4 °C (2)",
}


def _cell_key(c: str) -> int:
    return int("".join(ch for ch in c if ch.isdigit()))


def _panel_grid(n: int, ncols: int, w: float = 3.4, h: float = 2.5, sharey=True):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), sharey=sharey,
                             squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


# --------------------------------------------------------------------------- #
# Capacity / SoH trajectories
# --------------------------------------------------------------------------- #
def capacity_vs_cycle_ba(ba: pd.DataFrame) -> None:
    groups = list(BA_GROUP_LABELS)
    fig, axes = _panel_grid(len(groups), 5, w=3.2, h=2.6)
    for ax, grp in zip(axes, groups):
        g = ba[ba.protocol_group == grp]
        for k, (cell, d) in enumerate(sorted(g.groupby("cell"), key=lambda x: _cell_key(x[0]))):
            v = d[d.is_valid]
            ax.plot(v.cycle_number, v.capacity_ah_2v7, color=CATEGORICAL[k], lw=1.4, label=cell)
            bad = d[~d.is_valid & d.capacity_ah_2v7.notna()]
            ax.scatter(bad.cycle_number, bad.capacity_ah_2v7, marker="x", s=14, color=MUTED, lw=0.8,
                       zorder=3, label="_nolegend_")
        ax.axhline(BA_RATED_CAPACITY_AH * EOL_SOH, color=INK_2, lw=0.8, ls="--")
        ax.set_title(BA_GROUP_LABELS[grp], fontsize=9)
        ax.legend(loc="lower left", fontsize=7, ncol=2, handlelength=1)
        ax.set_xlabel("Discharge cycle")
    axes[0].set_ylabel("Capacity to 2.7 V (Ah)")
    axes[5].set_ylabel("Capacity to 2.7 V (Ah)")
    fig.suptitle("Battery Aging: discharge capacity vs cycle  (× = cycle failing quality checks, "
                 "dashed = 1.6 Ah = 80 % of rated)", x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    save(fig, OUT / "01_ba_capacity_vs_cycle.png")


def capacity_vs_efc_rw(rw: pd.DataFrame) -> None:
    groups = list(RW_GROUP_LABELS)
    fig, axes = _panel_grid(len(groups), 4, w=3.4, h=2.6)
    for ax, grp in zip(axes, groups):
        g = rw[rw.group == grp]
        for k, (cell, d) in enumerate(sorted(g.groupby("cell"), key=lambda x: _cell_key(x[0]))):
            v = d[d.is_valid]
            ax.plot(v.efc, v.capacity_ah, color=CATEGORICAL[k], lw=1.4, marker="o", ms=3, label=cell)
        ax.set_title(RW_GROUP_LABELS[grp], fontsize=9)
        ax.legend(loc="lower left", fontsize=7, ncol=2, handlelength=1)
        ax.set_xlabel("Equivalent full cycles")
    axes[0].set_ylabel("Reference capacity (Ah)")
    axes[4].set_ylabel("Reference capacity (Ah)")
    fig.suptitle("Randomized Usage: 1 A reference capacity vs equivalent full cycles",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    save(fig, OUT / "02_rw_capacity_vs_efc.png")


def soh_comparison(ba: pd.DataFrame, rw: pd.DataFrame) -> None:
    """Normalised SoH (vs. each cell's own initial capacity) - comparable shape across cells."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    sets = [("Battery Aging, 2 A CC 24 °C", ["B0005", "B0006", "B0007", "B0018", "B0036"]),
            ("Battery Aging, 4 A CC", ["B0029", "B0030", "B0031", "B0032", "B0033", "B0034"])]
    for ax, (title, cells) in zip(axes[:2], sets):
        for k, cell in enumerate(cells):
            d = ba[(ba.cell == cell) & ba.is_valid]
            ax.plot(d.cycle_number, d.soh_initial, color=CATEGORICAL[k], lw=1.4, label=cell)
        ax.set_title(title)
        ax.set_xlabel("Discharge cycle")
        ax.legend(fontsize=7, ncol=2)
    # RW: one line per experiment group (median across its 4 cells, binned by EFC).
    ax = axes[2]
    for k, (grp, label) in enumerate(RW_GROUP_LABELS.items()):
        d = rw[(rw.group == grp) & rw.is_valid]
        bins = np.arange(0, d.efc.max() + 50, 50)
        m = d.groupby(pd.cut(d.efc, bins), observed=True).soh.median()
        ax.plot([iv.mid for iv in m.index], m.values, color=CATEGORICAL[k], lw=1.4, label=label)
    ax.set_title("Randomized Usage, median per group")
    ax.set_xlabel("Equivalent full cycles")
    ax.legend(fontsize=6.5)
    axes[0].set_ylabel("Capacity / initial capacity")
    for ax in axes:
        ax.axhline(EOL_SOH, color=INK_2, lw=0.8, ls="--")
    fig.tight_layout()
    save(fig, OUT / "03_soh_comparison.png")


# --------------------------------------------------------------------------- #
# Voltage / current / temperature profiles
# --------------------------------------------------------------------------- #
def _profile_panels(series_by_age, xcol, cols, labels, title, path, norm_max):
    fig, axes = plt.subplots(len(cols), 1, figsize=(7.5, 2.0 * len(cols)), sharex=True)
    for age, d in series_by_age:
        color = SEQ_CMAP(age / norm_max)
        for ax, col in zip(axes, cols):
            ax.plot(d[xcol], d[col], color=color, lw=1.2)
    for ax, lab in zip(axes, labels):
        ax.set_ylabel(lab)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time since start of step (s)")
    sm = plt.cm.ScalarMappable(cmap=SEQ_CMAP, norm=plt.Normalize(0, norm_max))
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("Cycle number" if "cycle" in title.lower() else "Equivalent full cycles")
    cb.outline.set_visible(False)
    save(fig, path)


def profiles_ba(ops: pd.DataFrame, ba: pd.DataFrame, cell: str = "B0005") -> None:
    cyc = ba[(ba.cell == cell) & ba.is_valid]
    picks = cyc.iloc[np.linspace(0, len(cyc) - 1, 7).round().astype(int)]
    ts = pd.read_parquet(INTERIM_DIR / "ba_timeseries.parquet",
                         filters=[("cell", "==", cell)])
    for kind, op_col in (("discharge", "op_index"), ("charge", "charge_op_index")):
        series = []
        for r in picks.itertuples():
            op = getattr(r, op_col)
            if pd.isna(op):
                continue
            d = ts[ts.op_index == int(op)].copy()
            d["current"] = d.current_measured.abs()
            series.append((r.cycle_number, d))
        _profile_panels(series, "time_s", ["voltage_measured", "current", "temperature_measured"],
                        ["Voltage (V)", "|Current| (A)", "Temperature (°C)"],
                        f"{cell}: {kind} profiles across cycle life",
                        OUT / f"04_ba_{kind}_profiles_{cell}.png", cyc.cycle_number.max())


def profiles_rw(rw: pd.DataFrame, cell: str = "RW9") -> None:
    ts = pd.read_parquet(INTERIM_DIR / "rw_timeseries.parquet",
                         filters=[("cell", "==", cell), ("comment", "==", "reference discharge")])
    steps = pd.read_parquet(INTERIM_DIR / "rw_steps.parquet", filters=[("cell", "==", cell)])
    checks = rw[(rw.cell == cell) & rw.is_valid]
    picks = checks.iloc[np.linspace(0, len(checks) - 1, 7).round().astype(int)]
    series = [(r.efc, ts[ts.step_index == r.step_index]) for r in picks.itertuples()]
    _profile_panels(series, "relative_time_s", ["voltage", "current", "temperature"],
                    ["Voltage (V)", "Current (A)", "Temperature (°C)"],
                    f"{cell}: 1 A reference discharge across life",
                    OUT / f"05_rw_reference_discharge_{cell}.png", checks.efc.max())

    # Random-walk operation: first vs last 40 RW steps (charge and discharge).
    full = pd.read_parquet(INTERIM_DIR / "rw_steps.parquet", filters=[("cell", "==", cell)])
    rw_steps = full[full.comment.str.contains("random walk")]
    fig, axes = plt.subplots(3, 2, figsize=(11, 6), sharey="row")
    for j, (label, sel) in enumerate((("First 120 RW steps", rw_steps.head(120)),
                                      ("Last 120 RW steps", rw_steps.tail(120)))):
        t0 = sel.t_start_s.min()
        for col, ax, lab in (("v_mean", axes[0, j], "Mean voltage (V)"),
                             ("i_mean", axes[1, j], "Mean current (A)"),
                             ("t_mean", axes[2, j], "Temperature (°C)")):
            ax.step((sel.t_start_s - t0) / 60, sel[col], where="post", color=CATEGORICAL[0], lw=1)
            ax.set_ylabel(lab if j == 0 else "")
        axes[0, j].set_title(f"{cell}: {label.lower()} (+ = discharge)")
        axes[2, j].set_xlabel("Minutes")
    fig.tight_layout()
    save(fig, OUT / f"05_rw_random_walk_{cell}.png")
    del steps


def temperature_across_cycles(ba: pd.DataFrame, rw: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for k, cell in enumerate(["B0005", "B0006", "B0007", "B0018", "B0036"]):
        d = ba[(ba.cell == cell) & ba.is_valid]
        axes[0].plot(d.cycle_number, d.dis_t_max, color=CATEGORICAL[k], lw=1.2, label=cell)
        axes[1].plot(d.cycle_number, d.time_to_tmax_s / 60, color=CATEGORICAL[k], lw=1.2, label=cell)
    axes[0].set_title("Peak discharge temperature")
    axes[0].set_ylabel("°C")
    axes[1].set_title("Time to peak temperature")
    axes[1].set_ylabel("minutes")
    for ax in axes[:2]:
        ax.set_xlabel("Discharge cycle")
        ax.legend(fontsize=7)
    for k, (grp, label) in enumerate(RW_GROUP_LABELS.items()):
        d = rw[(rw.group == grp) & rw.is_valid]
        bins = np.arange(0, d.efc.max() + 50, 50)
        m = d.groupby(pd.cut(d.efc, bins), observed=True).interval_rw_t_mean.median()
        axes[2].plot([iv.mid for iv in m.index], m.values, color=CATEGORICAL[k], lw=1.4, label=label)
    axes[2].set_title("RW: mean cell temperature during RW use")
    axes[2].set_xlabel("Equivalent full cycles")
    axes[2].set_ylabel("°C")
    axes[2].legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, OUT / "06_temperature_across_cycles.png")


# --------------------------------------------------------------------------- #
# Data quality, distributions, correlations
# --------------------------------------------------------------------------- #
def missing_values(ba: pd.DataFrame, rw: pd.DataFrame) -> None:
    from src.features.targets import BA_FLAG_COLUMNS
    feats_ba = [f for g in BA_GROUPS.values() for f in g]
    feats_rw = [f for g in RW_GROUPS.values() for f in g]
    miss = pd.concat([
        ba[feats_ba].isna().mean().rename("missing_fraction").to_frame().assign(dataset="battery_aging"),
        rw[feats_rw].isna().mean().rename("missing_fraction").to_frame().assign(dataset="randomized_usage"),
    ])
    TABLES.mkdir(parents=True, exist_ok=True)
    miss.to_csv(TABLES / "missing_values.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    for ax, (name, feats, df) in zip(axes[:2], (("Battery Aging", feats_ba, ba),
                                                 ("Randomized Usage", feats_rw, rw))):
        m = df[feats].isna().mean().sort_values()
        ax.barh(m.index, m.values * 100, color=CATEGORICAL[0], height=0.65)
        ax.set_title(f"{name}: missing values per feature")
        ax.set_xlabel("% of rows missing")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", visible=False)
    ax = axes[2]
    flags = ba.groupby("cell")[BA_FLAG_COLUMNS].sum()
    flags = flags.loc[sorted(flags.index, key=_cell_key)]
    left = np.zeros(len(flags))
    for k, f in enumerate(BA_FLAG_COLUMNS):
        ax.barh(flags.index, flags[f], left=left, color=CATEGORICAL[k], height=0.7,
                label=f.replace("flag_", "").replace("_", " "), edgecolor="#fcfcfb", lw=0.6)
        left += flags[f].to_numpy()
    ax.invert_yaxis()
    ax.set_title("Battery Aging: discharge cycles failing each quality check")
    ax.set_xlabel("Number of cycles (a cycle can fail several checks)")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, OUT / "07_missing_values_and_quality_flags.png")


def feature_distributions(ba: pd.DataFrame, rw: pd.DataFrame) -> None:
    cohort = ba[ba.in_model_cohort & ba.is_valid]
    sets = [("ba", cohort, ["soh", "chg_cc_duration_s", "chg_cv_duration_s", "v_at_300s",
                            "t_to_3p8v_s", "t_rise_300s", "re_ohm", "rct_ohm", "dis_t_max",
                            "rest_before_h", "v_noise_std", "fade_slope_10"]),
            ("rw", rw[rw.is_valid], ["soh", "efc", "ref_chg_cc_duration_s", "ref_chg_cv_duration_s",
                                     "r_operational_ohm", "r_pulse_ohm", "interval_rw_i_mean",
                                     "interval_rw_t_mean", "interval_frac_hit_vmin", "ref_t_max",
                                     "interval_discharge_ah", "fade_slope_4"])]
    for tag, df, feats in sets:
        fig, axes = _panel_grid(len(feats), 4, w=3.0, h=2.1, sharey=False)
        for ax, f in zip(axes, feats):
            x = df[f].replace([np.inf, -np.inf], np.nan).dropna()
            ax.hist(x, bins=30, color=CATEGORICAL[0], edgecolor="#fcfcfb", lw=0.4)
            ax.set_title(f, fontsize=8.5)
            ax.grid(axis="x", visible=False)
        name = "Battery Aging (model cohort, valid cycles)" if tag == "ba" else "Randomized Usage"
        fig.suptitle(f"Feature distributions: {name}", x=0.01, ha="left", color=INK_2)
        fig.tight_layout()
        save(fig, OUT / f"08_feature_distributions_{tag}.png")


def correlations(ba: pd.DataFrame, rw: pd.DataFrame) -> None:
    cohort = ba[ba.in_model_cohort & ba.is_valid]
    sets = [("ba", cohort, ["soh"] + [f for g in ("condition", "charge", "early", "partial",
                                                  "impedance", "usage", "full")
                                      for f in BA_GROUPS[g]]),
            ("rw", rw[rw.is_valid], ["soh"] + [f for g in ("usage", "charge", "pulse", "full")
                                               for f in RW_GROUPS[g]])]
    for tag, df, feats in sets:
        c = df[feats].corr(method="spearman")
        c.to_csv(TABLES / f"spearman_correlations_{tag}.csv")
        fig, ax = plt.subplots(figsize=(0.36 * len(feats) + 2.5, 0.36 * len(feats) + 1.5))
        im = ax.imshow(c.values, cmap=DIV_CMAP, vmin=-1, vmax=1)
        ax.set_xticks(range(len(feats)), feats, rotation=90, fontsize=7)
        ax.set_yticks(range(len(feats)), feats, fontsize=7)
        ax.grid(False)
        for i in range(len(feats)):
            for j in range(len(feats)):
                if np.isfinite(c.values[i, j]) and abs(c.values[i, j]) >= 0.5 and i != j:
                    ax.text(j, i, f"{c.values[i, j]:.1f}", ha="center", va="center", fontsize=5.5,
                            color="#fcfcfb" if abs(c.values[i, j]) > 0.75 else "#0b0b0b")
        cb = fig.colorbar(im, ax=ax, shrink=0.6)
        cb.set_label("Spearman ρ")
        cb.outline.set_visible(False)
        ax.set_title("Battery Aging (model cohort)" if tag == "ba" else "Randomized Usage")
        save(fig, OUT / f"09_correlations_{tag}.png")


def ageing_relations(ba: pd.DataFrame, rw: pd.DataFrame) -> None:
    cohort = ba[ba.in_model_cohort & ba.is_valid]
    panels = [(cohort, "re_ohm", "Electrolyte resistance Re (Ω)", "Battery Aging"),
              (cohort, "rct_ohm", "Charge-transfer resistance Rct (Ω)", "Battery Aging"),
              (cohort, "chg_cc_duration_s", "CC charge duration (s)", "Battery Aging"),
              (cohort, "v_at_300s", "Voltage 300 s into discharge (V)", "Battery Aging"),
              (rw[rw.is_valid], "r_operational_ohm", "Resistance from RW steps (Ω)", "Randomized Usage"),
              (rw[rw.is_valid], "ref_chg_cc_duration_s", "CC reference-charge duration (s)",
               "Randomized Usage")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, (df, f, lab, name) in zip(axes.flat, panels):
        ycol = "soh" if name == "Battery Aging" else "soh"
        ax.scatter(df[f], df[ycol], s=8, color=CATEGORICAL[0], alpha=0.5, lw=0)
        rho = df[[f, ycol]].corr(method="spearman").iloc[0, 1]
        ax.set_title(f"{name}: ρ = {rho:.2f}", fontsize=9)
        ax.set_xlabel(lab)
        ax.set_ylabel("SoH")
    fig.tight_layout()
    save(fig, OUT / "10_ageing_relations.png")


def sensor_noise(ba: pd.DataFrame) -> None:
    n = ba.groupby("cell")[["v_noise_std", "t_noise_std"]].median()
    n = n.loc[sorted(n.index, key=_cell_key)]
    n.to_csv(TABLES / "sensor_noise_ba.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
    axes[0].bar(n.index, n.v_noise_std * 1000, color=CATEGORICAL[0], width=0.7)
    axes[0].set_ylabel("mV")
    axes[0].set_title("Estimated voltage noise during discharge (median per cell)")
    axes[1].bar(n.index, n.t_noise_std, color=CATEGORICAL[1], width=0.7)
    axes[1].set_ylabel("°C")
    axes[1].set_title("Estimated temperature noise during discharge")
    for ax in axes:
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, OUT / "11_sensor_noise_ba.png")


def run() -> None:
    apply_style()
    TABLES.mkdir(parents=True, exist_ok=True)
    ba = pd.read_parquet(PROCESSED_DIR / "ba_cycles.parquet")
    rw = pd.read_parquet(PROCESSED_DIR / "rw_checks.parquet")
    ops = pd.read_parquet(INTERIM_DIR / "ba_operations.parquet")
    capacity_vs_cycle_ba(ba)
    capacity_vs_efc_rw(rw)
    soh_comparison(ba, rw)
    profiles_ba(ops, ba)
    profiles_rw(rw)
    temperature_across_cycles(ba, rw)
    missing_values(ba, rw)
    feature_distributions(ba, rw)
    correlations(ba, rw)
    ageing_relations(ba, rw)
    sensor_noise(ba)
    # Plain-text summaries used in the README.
    ba[ba.in_model_cohort & ba.is_valid].groupby("cell").agg(
        cycles=("cycle_number", "size"), soh_first=("soh", "first"), soh_last=("soh", "last"),
        eol_cycle=("eol_cycle", "first")).to_csv(TABLES / "ba_cohort_summary.csv")


if __name__ == "__main__":
    run()
