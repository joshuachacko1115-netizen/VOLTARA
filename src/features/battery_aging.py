"""Per-discharge-cycle features for the Battery Aging dataset (B0005 ... B0056).

One output row = one discharge cycle of one cell, in cycle order. Columns are
grouped by what information they need (see FEATURE_GROUPS):

  condition  - test conditions known before the cycle (ambient temperature, load current)
  charge     - the charge that preceded this discharge (CC/CV durations, charge Ah, temps)
  early      - the first <= 600 s of the discharge (voltage at fixed times, temperature rise)
  partial    - time for the voltage to fall to 3.8 V / 3.6 V (part of the discharge curve)
  impedance  - most recent EIS result (Re, Rct) measured *before* this discharge
  usage      - cycle number, elapsed time, rest before the cycle
  full       - needs the complete discharge to 2.7 V (duration, energy, T max).
               These are almost the capacity itself, so they are excluded from SoH
               models and used only to demonstrate target leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BA_CAPACITY_CUTOFF_V, BA_CELL_INFO
from src.preprocessing.cleaning import interpolate_short_gaps, noise_std

LOAD_ON_A = 0.1           # current above which the load is considered applied
CV_END_A = 0.05           # charge considered finished below this current (README: 20 mA)
CC_END_V = 4.195          # CC -> CV transition (README: 4.2 V)

FEATURE_GROUPS = {
    "condition": ["ambient_temperature", "i_set"],
    "charge": ["chg_cc_duration_s", "chg_cv_duration_s", "chg_ah", "chg_v_start",
               "chg_t_mean", "chg_t_max"],
    "early": ["v_at_60s", "v_at_300s", "v_at_600s", "dis_t_start", "t_rise_300s"],
    "partial": ["t_to_3p8v_s", "t_to_3p6v_s"],
    "impedance": ["re_ohm", "rct_ohm", "impedance_age_cycles"],
    "usage": ["cycle_number", "elapsed_days", "rest_before_h"],
    "full": ["duration_to_cutoff_s", "energy_wh", "dis_v_mean", "dis_v_std",
             "dis_t_max", "dis_t_rise", "time_to_tmax_s"],
}


def _clean_signals(d: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, dict]:
    t = d["time_s"].to_numpy(float)
    out = {c: interpolate_short_gaps(t, d[c].to_numpy(float), max_gap=3) for c in cols}
    good = np.isfinite(t)
    for c in cols:
        good &= np.isfinite(out[c])
    return t[good], {c: v[good] for c, v in out.items()}


def _time_to_cross_below(t, v, level, start=0):
    """Interpolated time at which v first falls to <= level after index `start`."""
    below = np.flatnonzero(v[start:] <= level)
    if below.size == 0:
        return np.nan, None
    k = start + below[0]
    if k == start:
        return t[k], k
    t1, t2, v1, v2 = t[k - 1], t[k], v[k - 1], v[k]
    tc = t1 + (v1 - level) * (t2 - t1) / (v1 - v2) if v1 != v2 else t2
    return tc, k


def discharge_features(d: pd.DataFrame) -> dict:
    cols = ["voltage_measured", "current_measured", "temperature_measured"]
    n_raw = len(d)
    t, s = _clean_signals(d, cols)
    feats = {"n_samples": n_raw, "n_dropped_samples": n_raw - t.size}
    if t.size < 10:
        return feats
    v, i, temp = s["voltage_measured"], -s["current_measured"], s["temperature_measured"]
    on = np.flatnonzero(i > LOAD_ON_A)
    if on.size == 0:
        feats["i_set"] = 0.0
        return feats
    k0 = on[0]
    t0 = t[k0]
    rel = t - t0
    loaded = i > LOAD_ON_A
    feats["i_set"] = float(np.median(i[loaded]))
    feats["i_std"] = float(np.std(i[loaded]))

    # Capacity / energy to the common 2.7 V cutoff.
    tc, kc = _time_to_cross_below(t, v, BA_CAPACITY_CUTOFF_V, start=k0)
    feats["reached_cutoff"] = kc is not None
    if kc is not None:
        tt = np.append(t[k0:kc], tc)
        ii = np.append(i[k0:kc], np.interp(tc, t[kc - 1:kc + 1], i[kc - 1:kc + 1]) if kc > k0 else i[kc])
        vv = np.append(v[k0:kc], BA_CAPACITY_CUTOFF_V)
        ii = np.clip(ii, 0, None)
        feats["capacity_ah_2v7"] = float(np.trapezoid(ii, tt) / 3600)
        feats["energy_wh"] = float(np.trapezoid(ii * vv, tt) / 3600)
        feats["duration_to_cutoff_s"] = float(tc - t0)
        window = slice(k0, kc + 1)
    else:
        window = slice(k0, None)
    feats["ah_integrated_total"] = float(np.trapezoid(np.clip(i, 0, None), t) / 3600)

    # Early window: values at fixed times after the load is applied.
    for sec in (60, 300, 600):
        feats[f"v_at_{sec}s"] = float(np.interp(sec, rel[k0:], v[k0:])) if rel[-1] >= sec else np.nan
    feats["dis_t_start"] = float(temp[k0])
    feats["t_rise_300s"] = (float(np.interp(300, rel[k0:], temp[k0:]) - temp[k0])
                            if rel[-1] >= 300 else np.nan)

    # Partial curve.
    for level, name in ((3.8, "t_to_3p8v_s"), (3.6, "t_to_3p6v_s")):
        tl, _ = _time_to_cross_below(t, v, level, start=k0)
        feats[name] = float(tl - t0) if np.isfinite(tl) else np.nan

    # Full-discharge statistics (leakage-prone, see module docstring).
    vw, tw = v[window], temp[window]
    feats["dis_v_mean"] = float(vw.mean())
    feats["dis_v_std"] = float(vw.std())
    feats["dis_v_min"] = float(v[k0:].min())
    feats["dis_t_max"] = float(tw.max())
    feats["dis_t_rise"] = float(tw.max() - temp[k0])
    feats["time_to_tmax_s"] = float(t[window][np.argmax(tw)] - t0)

    # Sensor-noise estimates on the loaded part of the discharge.
    feats["v_noise_std"] = noise_std(v[loaded])
    feats["t_noise_std"] = noise_std(temp[loaded])
    return feats


def charge_features(d: pd.DataFrame) -> dict:
    t, s = _clean_signals(d, ["voltage_measured", "current_measured", "temperature_measured"])
    if t.size < 10:
        return {}
    v, i, temp = s["voltage_measured"], s["current_measured"], s["temperature_measured"]
    on = np.flatnonzero(i > 0.5)
    if on.size == 0:
        return {}
    k0 = on[0]
    feats = {"chg_v_start": float(v[k0]), "chg_t_mean": float(temp.mean()),
             "chg_t_max": float(temp.max()),
             "chg_ah": float(np.trapezoid(np.clip(i, 0, None), t) / 3600)}
    cv = np.flatnonzero(v[k0:] >= CC_END_V)
    if cv.size:
        kcv = k0 + cv[0]
        feats["chg_cc_duration_s"] = float(t[kcv] - t[k0])
        end = np.flatnonzero(i[kcv:] < CV_END_A)
        feats["chg_cv_duration_s"] = float(t[kcv + end[0]] - t[kcv]) if end.size else np.nan
        feats["chg_complete"] = bool(end.size)
    else:
        feats["chg_complete"] = False
    return feats


def build_cycle_table(operations: pd.DataFrame, timeseries: pd.DataFrame) -> pd.DataFrame:
    """One row per discharge cycle with all feature groups (no cleaning, no targets)."""
    groups = {k: g for k, g in timeseries.groupby(["cell", "op_index"], sort=False)}
    rows = []
    for cell, ops in operations.groupby("cell", sort=False):
        ops = ops.sort_values("op_index").reset_index(drop=True)
        ops["end_time"] = ops.start_time + pd.to_timedelta(ops.duration_s.fillna(0), unit="s")
        t_first = ops.start_time.min()
        last_charge = None
        last_imp = {"re_ohm": np.nan, "rct_ohm": np.nan, "cycle": None}
        prev_end = None
        n_dis = 0
        for op in ops.itertuples():
            if op.type == "impedance":
                if np.isfinite(op.re_ohm) and np.isfinite(op.rct_ohm):
                    last_imp = {"re_ohm": op.re_ohm, "rct_ohm": op.rct_ohm, "cycle": n_dis}
            elif op.type == "charge":
                last_charge = op.op_index
            elif op.type == "discharge":
                n_dis += 1
                row = {"cell": cell, "op_index": op.op_index, "cycle_number": n_dis,
                       "start_time": op.start_time, "ambient_temperature": op.ambient_temperature,
                       "capacity_ah_stored": op.capacity_ah,
                       "elapsed_days": (op.start_time - t_first).total_seconds() / 86400,
                       "rest_before_h": ((op.start_time - prev_end).total_seconds() / 3600
                                         if prev_end is not None else np.nan)}
                info = BA_CELL_INFO.get(cell, {})
                row.update({"cutoff_v": info.get("cutoff_v"), "profile": info.get("profile"),
                            "protocol_group": info.get("protocol_group")})
                if (cell, op.op_index) in groups:
                    row.update(discharge_features(groups[(cell, op.op_index)]))
                if last_charge is not None and (cell, last_charge) in groups:
                    row.update(charge_features(groups[(cell, last_charge)]))
                    row["charge_op_index"] = last_charge
                last_charge = None  # a charge belongs to the next discharge only
                row["re_ohm"], row["rct_ohm"] = last_imp["re_ohm"], last_imp["rct_ohm"]
                row["impedance_age_cycles"] = (n_dis - last_imp["cycle"]
                                               if last_imp["cycle"] is not None else np.nan)
                rows.append(row)
            prev_end = op.end_time
    return pd.DataFrame(rows)
