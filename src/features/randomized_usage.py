"""Per-reference-check features for the Randomized Usage dataset (RW1 ... RW28).

The batteries spend most of their life in random-walk (RW) operation; every
~50 RW cycles (or 1500 RW steps for RW9-RW12) a block of reference tests is run.
One output row = one such *reference check* of one cell.

  capacity  - median Ah of the valid 1 A reference discharges in the check
  usage     - what happened since the previous check: discharged Ah, mean RW
              current and temperature, an operational resistance estimate
  charge    - CC / CV durations of the 2 A reference charge
  pulse     - resistance from the 1 A pulsed-load test (ffilled to later checks)
  full      - energy and duration of the reference discharge itself (leakage-prone)

The ageing axis is equivalent full cycles (EFC) = cumulative discharged Ah divided
by the cell's first reference capacity, because "cycle number" is not defined for
random-walk operation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RW_REFERENCE_CURRENT_A

FEATURE_GROUPS = {
    "condition": ["ambient_40c", "dist_skewed_high", "dist_skewed_low", "dist_uniform"],
    "usage": ["efc", "elapsed_days", "interval_discharge_ah", "interval_rw_i_mean",
              "interval_rw_t_mean", "interval_rw_t_max", "interval_frac_hit_vmin",
              "r_operational_ohm"],
    "charge": ["ref_chg_cc_duration_s", "ref_chg_cv_duration_s", "ref_chg_t_max"],
    "pulse": ["r_pulse_ohm"],
    "full": ["ref_energy_wh", "ref_duration_s", "ref_v_mean", "ref_t_max"],
}

RW_DISCHARGE = "discharge (random walk)"
RW_REST = "rest (random walk)"


def _valid_reference(s: pd.DataFrame) -> pd.Series:
    """Reference discharges that ran at ~1 A from near-full to the 3.2 V cutoff."""
    return ((s.comment == "reference discharge")
            & ((s.i_mean - RW_REFERENCE_CURRENT_A).abs() < 0.1)
            & (s.duration_s > 1800)
            & (s.v_max >= 4.0)
            & (s.v_end <= 3.25))


def _group_flags(group: str) -> dict:
    g = group.lower()
    return {"ambient_40c": int("40c" in g),
            "dist_skewed_high": int("skewed_high" in g),
            "dist_skewed_low": int("skewed_low" in g),
            "dist_uniform": int("uniform" in g)}


def _operational_resistance(rw: pd.DataFrame) -> pd.Series:
    """(V at end of RW rest - V at start of next RW discharge) / current, per step."""
    prev_rest = rw.comment.shift(1).eq(RW_REST)
    dv = rw.v_end.shift(1) - rw.v_start
    ok = rw.comment.eq(RW_DISCHARGE) & prev_rest & (rw.i_mean > 0.5)
    return (dv / rw.i_mean).where(ok)


def _pulse_resistance(s: pd.DataFrame) -> pd.Series:
    prev_rest = s.comment.shift(1).eq("pulsed load (rest)")
    ok = s.comment.eq("pulsed load (discharge)") & prev_rest & (s.i_mean > 0.5)
    return ((s.v_end.shift(1) - s.v_start) / s.i_mean).where(ok)


def _charge_features(d: pd.DataFrame) -> dict:
    """CC/CV durations of a 2 A reference charge (current is negative when charging)."""
    t, v, i = d.relative_time_s.to_numpy(), d.voltage.to_numpy(), -d.current.to_numpy()
    out = {"ref_chg_t_max": float(np.nanmax(d.temperature)) if d.temperature.notna().any() else np.nan}
    cc = np.flatnonzero(i > 1.5)
    cv = np.flatnonzero(v >= 4.195)
    if cc.size and cv.size and cv[0] >= cc[0]:
        out["ref_chg_cc_duration_s"] = float(t[cv[0]] - t[cc[0]])
        out["ref_chg_cv_duration_s"] = float(t[-1] - t[cv[0]])
    return out


def build_check_table(steps: pd.DataFrame, ref_series: pd.DataFrame) -> pd.DataFrame:
    rows = []
    charge_series = {k: g for k, g in ref_series[ref_series.comment == "reference charge"]
                     .groupby(["cell", "step_index"], sort=False)}
    for cell, s in steps.groupby("cell", sort=False):
        s = s.sort_values("step_index").reset_index(drop=True)
        group = s.group.iloc[0]
        s["valid_ref"] = _valid_reference(s)
        s["r_op"] = _operational_resistance(s)
        s["r_pulse"] = _pulse_resistance(s)
        is_rw = s.comment.str.contains("random walk", regex=False)

        # A new check starts at a reference discharge with RW steps since the previous one.
        ref_rows = s.index[s.comment == "reference discharge"]
        check_of_row = pd.Series(np.nan, index=s.index)
        check_id, last_ref = -1, None
        for r in ref_rows:
            if last_ref is None or is_rw.iloc[last_ref + 1:r].any():
                check_id += 1
            check_of_row[r] = check_id
            last_ref = r
        s["check_id"] = check_of_row.ffill()  # steps after a check belong to it

        cum_dis = s.ah.where((s.type == "D") & (s.ah > 0), 0).cumsum()
        t0 = s.date.min()
        prev_start = 0
        last_pulse_r = np.nan
        for cid in range(check_id + 1):
            block = s[s.check_id == cid]
            refs = block[block.comment == "reference discharge"]
            valid = refs[refs.valid_ref]
            first = refs.index[0]
            interval = s.loc[prev_start:first - 1]
            rw_d = interval[interval.comment == RW_DISCHARGE]
            w = rw_d.duration_s.clip(lower=0)
            # 15 W discharge: drop aborted steps (e.g. RW17 has one at 0 A for 300 s).
            power = block[(block.comment == "reference power discharge") & (block.i_mean > 3.0)
                          & (block.v_end <= 3.25)]
            pulses = block.r_pulse.dropna()
            if len(pulses):
                last_pulse_r = float(pulses.iloc[:3].median())  # first pulses = high SOC
            row = {
                "cell": cell, "group": group, "check_id": cid,
                "step_index": int(s.step_index[first]), "date": s.date[first],
                "elapsed_days": (s.date[first] - t0).total_seconds() / 86400,
                "n_ref_discharges": len(refs), "n_valid_ref_discharges": len(valid),
                "capacity_ah": float(valid.ah.median()) if len(valid) else np.nan,
                "capacity_spread_ah": float(np.ptp(valid.ah)) if len(valid) > 1 else 0.0,
                "ref_energy_wh": float(valid.wh.median()) if len(valid) else np.nan,
                "ref_duration_s": float(valid.duration_s.median()) if len(valid) else np.nan,
                "ref_v_mean": float(valid.v_mean.median()) if len(valid) else np.nan,
                "ref_t_max": float(valid.t_max.median()) if len(valid) else np.nan,
                "ref_n_bad_temperature": int(refs.n_bad_temperature.sum()),
                "cum_discharge_ah": float(cum_dis[first - 1]) if first > 0 else 0.0,
                "interval_discharge_ah": float(interval.ah.where((interval.type == "D") & (interval.ah > 0)).sum()),
                "interval_n_rw_steps": len(rw_d),
                "interval_rw_i_mean": float(np.average(rw_d.i_mean, weights=w)) if w.sum() > 0 else np.nan,
                "interval_rw_t_mean": float(np.average(rw_d.t_mean.fillna(rw_d.t_mean.mean()), weights=w))
                                      if w.sum() > 0 and rw_d.t_mean.notna().any() else np.nan,
                "interval_rw_t_max": float(rw_d.t_max.max()) if len(rw_d) else np.nan,
                "interval_frac_hit_vmin": float((rw_d.v_end <= 3.21).mean()) if len(rw_d) else np.nan,
                "interval_n_bad_temperature": int(interval.n_bad_temperature.sum()),
                "r_operational_ohm": float(interval.r_op.median()) if interval.r_op.notna().any() else np.nan,
                "r_pulse_ohm": last_pulse_r,
                "power_discharge_duration_s": float(power.duration_s.median()) if len(power) else np.nan,
                # Rest voltage after the 15 W step: > 4.0 V means it hit 3.2 V while nearly full,
                # i.e. it was limited by power (IR drop), not by stored energy.
                "power_discharge_rest_v_after": float(s.v_end.reindex(power.index + 1).median())
                                                if len(power) else np.nan,
            }
            row.update(_group_flags(group))
            # Only a reference charge that starts from the discharged state has
            # comparable CC/CV durations: it must follow a valid reference discharge
            # within two steps and return >= 90 % of that discharge's Ah. (Some
            # reference charges start from a nearly full cell, ~4.1 V.)
            for r in valid.index:
                nxt = s.loc[r + 1:r + 2]
                chg = nxt[nxt.comment == "reference charge"]
                if len(chg) and -chg.ah.iloc[0] >= 0.9 * s.ah[r]:
                    key = (cell, int(chg.step_index.iloc[0]))
                    if key in charge_series:
                        row.update(_charge_features(charge_series[key]))
                        break
            rows.append(row)
            prev_start = first
    out = pd.DataFrame(rows)
    # First check without an interval has no RW usage yet.
    first_c0 = out.groupby("cell").capacity_ah.transform(lambda c: c.dropna().iloc[0] if c.notna().any() else np.nan)
    out["capacity_initial_ah"] = first_c0
    out["efc"] = out.cum_discharge_ah / first_c0
    return out
