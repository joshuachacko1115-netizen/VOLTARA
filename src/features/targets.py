"""Quality flags, SoH / RUL targets and causal trend features.

Target definitions
  Battery Aging      SoH = capacity to 2.7 V / 2.0 Ah (rated capacity from the README)
  Randomized Usage   SoH = reference capacity / the cell's first valid reference capacity
  End of life (EOL)  first point where the *centred* rolling median of SoH <= EOL_SOH.
                     The centred median stops a single noisy cycle from defining EOL;
                     it uses future values, which is fine for a label but means it
                     must never be used as an input feature.
  RUL                Battery Aging: EOL cycle - current cycle (cycles)
                     Randomized Usage: EFC at EOL - current EFC (equivalent full cycles),
                     with the EOL EFC linearly interpolated between the two checks
                     either side of the crossing.
  Censoring          Cells that never reach EOL have no RUL label (NaN). They get
                     `rul_lower_bound` (remaining observed life) and are left out of
                     RUL training and scoring, because treating "end of test" as EOL
                     would systematically understate their life.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BA_CELL_INFO, BA_MODEL_CELLS, BA_RATED_CAPACITY_AH, EOL_SOH
from src.preprocessing.cleaning import (causal_initial, causal_rolling, causal_rolling_slope, hampel_flags,
                                        residual_vs_recent)

MIN_EOL_CYCLE = 10

BA_FLAG_COLUMNS = ["flag_no_load", "flag_incomplete", "flag_current_off_nominal",
                   "flag_low_capacity", "flag_capacity_mismatch", "flag_outlier"]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def eol_position(x: np.ndarray, soh: np.ndarray, threshold: float = EOL_SOH,
                 window: int = 5, interpolate: bool = False) -> float:
    """Ageing-axis value (cycle or EFC) at which smoothed SoH first reaches threshold."""
    sm = pd.Series(soh).rolling(window, center=True, min_periods=1).median().to_numpy()
    hit = np.flatnonzero(sm <= threshold)
    if hit.size == 0:
        return np.nan
    k = hit[0]
    if not interpolate or k == 0:
        return float(x[k])
    x1, x2, s1, s2 = x[k - 1], x[k], sm[k - 1], sm[k]
    return float(x1 + (s1 - threshold) * (x2 - x1) / (s1 - s2)) if s1 != s2 else float(x2)


def add_rul(df: pd.DataFrame, axis: str, eol_col: str, out: str) -> pd.DataFrame:
    df[out] = np.where(df[axis] <= df[eol_col], df[eol_col] - df[axis], np.nan)
    df["post_eol"] = df[axis] > df[eol_col]
    last = df.groupby("cell")[axis].transform("max")
    df["censored"] = df[eol_col].isna()
    df["rul_lower_bound"] = np.where(df.censored, last - df[axis], np.nan)
    return df


# --------------------------------------------------------------------------- #
# Battery Aging
# --------------------------------------------------------------------------- #
def ba_quality_flags(c: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    nominal = c.cell.map(lambda x: BA_CELL_INFO.get(x, {}).get("nominal_current_a", np.nan))
    cap = c.capacity_ah_2v7
    c["flag_no_load"] = c.i_set.fillna(0) < 0.5
    c["flag_incomplete"] = ~c.reached_cutoff.fillna(False).astype(bool)
    c["flag_current_off_nominal"] = (c.profile == "cc") & ((c.i_set - nominal).abs() > 0.3)
    c["flag_low_capacity"] = cap < 0.3
    c["flag_capacity_mismatch"] = (cap - c.capacity_ah_stored).abs() > 0.05
    pre = ~c[BA_FLAG_COLUMNS[:-1]].any(axis=1) & cap.notna()
    c["flag_outlier"] = False
    for cell, idx in c[pre].groupby("cell").groups.items():
        x = cap[idx].to_numpy()
        # min_scale 0.02 Ah keeps genuine capacity regeneration after long rests
        # (jumps of a few hundredths of an Ah) from being flagged as outliers.
        spikes = hampel_flags(x, window=7, n_sigma=3.0, min_scale=0.02)
        # A 7-point window cannot catch runs of several bad cycles (e.g. B0033 around
        # cycle 140 drops to ~0.3 Ah for ~8 cycles and then recovers), so also flag
        # cycles far below a wide centred median.
        wide = pd.Series(x).rolling(21, center=True, min_periods=1).median().to_numpy()
        c.loc[idx, "flag_outlier"] = spikes | (x < 0.75 * wide)
    c["is_valid"] = pre & ~c.flag_outlier
    return c


def ba_targets_and_trends(c: pd.DataFrame) -> pd.DataFrame:
    c = c.sort_values(["cell", "cycle_number"]).reset_index(drop=True)
    c["soh"] = c.capacity_ah_2v7 / BA_RATED_CAPACITY_AH
    c["in_model_cohort"] = c.cell.isin(BA_MODEL_CELLS)
    for col in ["soh_initial", "eol_cycle", "cap_roll_med5", "soh_roll_med5", "cap_roll_std5",
                "fade_slope_10", "fade_slope_20", "capacity_fade_ah", "cap_resid_prev5"]:
        c[col] = np.nan
    for cell, g in c.groupby("cell"):
        v = g[g.is_valid]
        if v.empty:
            continue
        cap = v.capacity_ah_2v7.to_numpy()
        cyc = v.cycle_number.to_numpy(float)
        c0 = np.median(cap[:3])
        c.loc[v.index, "soh_initial"] = cap / c0
        c.loc[g.index, "eol_cycle"] = eol_position(cyc, v.soh.to_numpy())
        # Causal trends: computed on valid cycles only, each value uses the past only.
        med5 = causal_rolling(cap, 5, "median")
        c.loc[v.index, "cap_roll_med5"] = med5
        c.loc[v.index, "soh_roll_med5"] = med5 / BA_RATED_CAPACITY_AH
        c.loc[v.index, "cap_roll_std5"] = causal_rolling(cap, 5, "std")
        # Slopes in Ah per 100 cycles (negative = fading).
        c.loc[v.index, "fade_slope_10"] = 100 * causal_rolling_slope(cap, cyc, window=10)
        c.loc[v.index, "fade_slope_20"] = 100 * causal_rolling_slope(cap, cyc, window=20)
        # Causal reference: rows 0-1 must not see cycles 2-3 (soh_initial above is descriptive only).
        c.loc[v.index, "capacity_fade_ah"] = causal_initial(cap, 3) - med5
        c.loc[v.index, "cap_resid_prev5"] = residual_vs_recent(cap, 5)
    c = add_rul(c, "cycle_number", "eol_cycle", "rul_cycles")
    c.loc[~c.is_valid, "rul_cycles"] = np.nan
    # Cells that are at the EOL threshold almost from the start (B0033/B0034 at 4 A:
    # the higher rate alone gives ~1.6 Ah) have no meaningful RUL trajectory.
    c["in_rul_cohort"] = (c.in_model_cohort & ~c.censored
                          & (c.eol_cycle >= MIN_EOL_CYCLE))
    return c


# --------------------------------------------------------------------------- #
# Randomized Usage
# --------------------------------------------------------------------------- #
def rw_targets_and_trends(c: pd.DataFrame) -> pd.DataFrame:
    c = c.sort_values(["cell", "check_id"]).reset_index(drop=True)
    c["is_valid"] = c.capacity_ah.notna() & c.efc.notna()
    c["soh"] = c.capacity_ah / c.capacity_initial_ah
    for col in ["eol_efc", "soh_roll_med3", "fade_slope_4", "fade_slope_8", "cap_resid_prev3"]:
        c[col] = np.nan
    for cell, g in c.groupby("cell"):
        v = g[g.is_valid]
        if v.empty:
            continue
        soh, efc = v.soh.to_numpy(), v.efc.to_numpy()
        c.loc[g.index, "eol_efc"] = eol_position(efc, soh, window=3, interpolate=True)
        c.loc[v.index, "soh_roll_med3"] = causal_rolling(soh, 3, "median")
        # SoH change per 100 EFC over the last 4 / 8 checks (negative = fading).
        c.loc[v.index, "fade_slope_4"] = 100 * causal_rolling_slope(soh, efc, window=4)
        c.loc[v.index, "fade_slope_8"] = 100 * causal_rolling_slope(soh, efc, window=8)
        c.loc[v.index, "cap_resid_prev3"] = residual_vs_recent(soh, 3)
    c = add_rul(c, "efc", "eol_efc", "rul_efc")
    c.loc[~c.is_valid, "rul_efc"] = np.nan
    # r_pulse is only measured in some checks; carry the last measurement forward (causal).
    c["r_pulse_ohm"] = c.groupby("cell").r_pulse_ohm.ffill()
    return c
