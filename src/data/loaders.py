"""Parse the NASA .mat files into tidy pandas tables.

The two datasets have different MATLAB layouts, so each has its own loader.

Battery Aging (B0005 ... B0056), variable named after the cell, e.g. "B0005":
    B0005.cycle[k].type                 'charge' | 'discharge' | 'impedance'
    B0005.cycle[k].ambient_temperature  deg C
    B0005.cycle[k].time                 MATLAB date vector of the start time
    B0005.cycle[k].data                 measurement struct (fields depend on type)
  Notes found while inspecting the files (they differ from the README):
    * discharge cycles use Current_load / Voltage_load, not Current_charge / Voltage_charge;
    * discharge current is negative (sign convention: + = into the battery);
    * impedance cycles use 'Rectified_Impedance' (capital I).

Randomized Usage (RW1 ... RW28), variable "data":
    data.procedure, data.description    text
    data.step[k].comment                e.g. 'reference discharge', 'discharge (random walk)'
    data.step[k].type                   'C' charge | 'D' discharge | 'R' rest
    data.step[k].time / relativeTime    s from experiment start / step start
    data.step[k].voltage, current, temperature
    data.step[k].date                   'dd-Mon-yyyy HH:MM:SS'
  Here discharge current is positive.

Each loader returns plain DataFrames; build_dataset.py writes them to parquet.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

# Temperatures outside this range are physically impossible for these tests and
# appear as sensor drop-outs (e.g. about -4094 degC in RW2, RW3 and RW18).
TEMP_VALID_RANGE = (-30.0, 100.0)
# Plausible range for the fitted Re / Rct of an 18650 cell, in Ohm.
IMPEDANCE_VALID_RANGE = (0.0, 1.0)

# RW step comments whose raw samples are kept (the random-walk steps themselves
# are only summarised, to keep the processed data a manageable size).
RW_KEEP_RAW_COMMENTS = {
    "reference charge",
    "reference discharge",
    "reference power discharge",
    "rest post reference discharge",
    "pulsed load (discharge)",
    "pulsed load (rest)",
}


def _arr(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float)).ravel()


def matlab_datevec_to_datetime(vec) -> pd.Timestamp:
    v = _arr(vec)
    if v.size < 6 or not np.isfinite(v).all():
        return pd.NaT
    y, mo, d, h, mi, s = v[:6]
    return pd.Timestamp(datetime(int(y), int(mo), int(d), int(h), int(mi)) + timedelta(seconds=float(s)))


def mask_invalid_temperature(t: np.ndarray) -> np.ndarray:
    t = t.copy()
    t[(t < TEMP_VALID_RANGE[0]) | (t > TEMP_VALID_RANGE[1])] = np.nan
    return t


# --------------------------------------------------------------------------- #
# Battery Aging dataset
# --------------------------------------------------------------------------- #
BA_SIGNALS = {
    # column name            charge field        discharge field
    "voltage_measured":     ("Voltage_measured", "Voltage_measured"),
    "current_measured":     ("Current_measured", "Current_measured"),
    "temperature_measured": ("Temperature_measured", "Temperature_measured"),
    "current_aux":          ("Current_charge", "Current_load"),
    "voltage_aux":          ("Voltage_charge", "Voltage_load"),
}


def load_battery_aging_cell(path: Path | str) -> dict[str, pd.DataFrame]:
    """Return {'operations': one row per cycle, 'timeseries': charge/discharge samples}."""
    path = Path(path)
    cell = path.stem
    mat = sio.loadmat(path, simplify_cells=True)
    cycles = mat[cell]["cycle"]
    if isinstance(cycles, dict):
        cycles = [cycles]

    ops, series = [], []
    type_counter: dict[str, int] = {}
    for op_index, c in enumerate(cycles):
        op_type = str(c["type"]).strip()
        type_index = type_counter.get(op_type, 0)
        type_counter[op_type] = type_index + 1
        d = c["data"]
        row = {
            "cell": cell,
            "op_index": op_index,
            "type": op_type,
            "type_index": type_index,
            "ambient_temperature": float(c["ambient_temperature"]),
            "start_time": matlab_datevec_to_datetime(c["time"]),
            "n_samples": 0,
            "duration_s": np.nan,
            "capacity_ah": np.nan,
            "re_ohm": np.nan,
            "rct_ohm": np.nan,
        }
        if op_type == "impedance":
            # Some fits in B0049-B0052 are complex or absurd (e.g. -9.7e14 Ohm);
            # those are treated as failed estimates, not as real resistances.
            invalid = False
            for key, col in (("Re", "re_ohm"), ("Rct", "rct_ohm")):
                v = np.atleast_1d(np.asarray(d.get(key, np.nan))).ravel()
                ok = (v.size == 1 and not np.iscomplexobj(v) and np.isfinite(v[0])
                      and IMPEDANCE_VALID_RANGE[0] < v[0] < IMPEDANCE_VALID_RANGE[1])
                row[col] = float(v[0]) if ok else np.nan
                invalid |= not ok
            row["impedance_fit_invalid"] = invalid
            ops.append(row)
            continue

        t = _arr(d.get("Time", []))
        row["n_samples"] = int(t.size)
        if t.size > 1:
            row["duration_s"] = float(t[-1] - t[0])
        if op_type == "discharge":
            cap = _arr(d.get("Capacity", np.nan))
            row["capacity_ah"] = float(cap[0]) if cap.size == 1 else np.nan
        ops.append(row)

        if t.size == 0:
            continue
        idx = 0 if op_type == "charge" else 1
        frame = {"cell": cell, "op_index": op_index, "type": op_type, "time_s": t}
        for col, fields in BA_SIGNALS.items():
            v = _arr(d.get(fields[idx], np.full(t.size, np.nan)))
            frame[col] = v if v.size == t.size else np.full(t.size, np.nan)
        frame["temperature_measured"] = mask_invalid_temperature(frame["temperature_measured"])
        series.append(pd.DataFrame(frame))

    operations = pd.DataFrame(ops)
    timeseries = pd.concat(series, ignore_index=True) if series else pd.DataFrame()
    return {"operations": operations, "timeseries": timeseries}


# --------------------------------------------------------------------------- #
# Randomized Usage dataset
# --------------------------------------------------------------------------- #
def _summarise_step(t, rt, v, i, temp) -> dict:
    n = rt.size
    out = {
        "n_samples": n,
        "t_start_s": t[0] if n else np.nan,
        "t_end_s": t[-1] if n else np.nan,
        "duration_s": rt[-1] - rt[0] if n > 1 else 0.0,
        "ah": np.nan, "wh": np.nan,
        "i_mean": np.nan, "i_abs_max": np.nan,
        "v_start": np.nan, "v_end": np.nan, "v_min": np.nan, "v_max": np.nan, "v_mean": np.nan,
        "t_mean": np.nan, "t_min": np.nan, "t_max": np.nan,
        "n_bad_temperature": 0,
    }
    if n == 0:
        return out
    out.update(i_mean=float(i.mean()), i_abs_max=float(np.abs(i).max()),
               v_start=float(v[0]), v_end=float(v[-1]), v_min=float(v.min()),
               v_max=float(v.max()), v_mean=float(v.mean()))
    if n > 1:
        # Signed charge/energy throughput; + = discharge in this dataset.
        out["ah"] = float(np.trapezoid(i, rt) / 3600.0)
        out["wh"] = float(np.trapezoid(i * v, rt) / 3600.0)
    good = np.isfinite(temp)
    out["n_bad_temperature"] = int((~good).sum())
    if good.any():
        out.update(t_mean=float(temp[good].mean()), t_min=float(temp[good].min()),
                   t_max=float(temp[good].max()))
    return out


def load_randomized_usage_cell(path: Path | str, group: str = "") -> dict:
    """Return {'steps': one row per step, 'timeseries': raw samples of reference steps,
    'procedure': str, 'description': str}."""
    path = Path(path)
    cell = path.stem
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["data"]
    steps = np.atleast_1d(data.step)

    rows, series = [], []
    for k, s in enumerate(steps):
        t, rt = _arr(s.time), _arr(s.relativeTime)
        v, i = _arr(s.voltage), _arr(s.current)
        temp = mask_invalid_temperature(_arr(s.temperature))
        if not (t.size == rt.size == v.size == i.size == temp.size):
            n = min(t.size, rt.size, v.size, i.size, temp.size)
            t, rt, v, i, temp = t[:n], rt[:n], v[:n], i[:n], temp[:n]
        comment = str(s.comment).strip()
        row = {"cell": cell, "group": group, "step_index": k, "comment": comment,
               "type": str(s.type).strip(),
               "date": pd.to_datetime(str(s.date), format="%d-%b-%Y %H:%M:%S", errors="coerce")}
        row.update(_summarise_step(t, rt, v, i, temp))
        rows.append(row)
        if comment in RW_KEEP_RAW_COMMENTS and rt.size:
            series.append(pd.DataFrame({
                "cell": cell, "step_index": k, "comment": comment,
                "time_s": t, "relative_time_s": rt,
                "voltage": v, "current": i, "temperature": temp,
            }))

    return {
        "steps": pd.DataFrame(rows),
        "timeseries": pd.concat(series, ignore_index=True) if series else pd.DataFrame(),
        "procedure": str(data.procedure),
        "description": str(data.description),
    }
