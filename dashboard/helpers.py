"""Presentation helpers for the Streamlit dashboard.

Everything that predicts goes through src.inference.predict.VoltaraPredictor; this
module only reads user input, summarises the predictor's output for display and
builds charts. No feature engineering or model logic lives here.
"""
from __future__ import annotations

import hashlib
import io
import re

import altair as alt
import numpy as np
import pandas as pd

from src.config import EOL_SOH
from src.inference.predict import EOL_NOTE, INPUT_DESCRIPTIONS, VoltaraPredictor
from src.models.range_estimation import RangeAssumptions
from src.visualization.style import CATEGORICAL, GRID, INK_2, MUTED, RISK_COLORS, STATUS

DATASET_LABELS = {
    "battery_aging": "Battery Aging (one row per discharge cycle)",
    "randomized_usage": "Randomized Usage (one row per reference check)",
}
AGEING_LABELS = {"cycle_number": "Discharge cycle", "efc": "Equivalent full cycles (EFC)"}
RISK_ICONS = {"Low": "●", "Medium": "▲", "High": "■"}
# An example must reach end of life no earlier than this (cycles or EFC); excludes B0033/B0034, which are
# below 80 % SoH from the start at their 4 A discharge rate. Same threshold as the RUL training cohort.
MIN_LIFE = 10
LEVELS = ["Low", "Medium", "High"]


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def read_history_csv(data: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV. Raises ValueError with a readable message."""
    for encoding in ("utf-8-sig", "cp1252"):  # UTF-8 (with Excel's byte-order mark), then Windows Excel
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("could not read the file: it is not a text CSV (save it as CSV UTF-8)")
    if "\x00" in text or sum(not (ch.isprintable() or ch in "\r\n\t") for ch in text[:2000]) > 20:
        raise ValueError("could not read the file: it does not look like a text CSV")
    if not text.strip():
        raise ValueError("the CSV file is empty")
    sep, decimal = _sniff_format(text)
    try:
        # round_trip: the default parser can change the last float bit, enough to flip a tree split.
        df = pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal, float_precision="round_trip")
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise ValueError(f"could not read the CSV file: {e}") from e
    if df.empty:
        raise ValueError("the CSV file has no rows")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _sniff_format(text: str) -> tuple[str, str]:
    """Field separator (, ; tab) from the header line, and decimal comma for ';' files that use it."""
    header = text.splitlines()[0]
    sep = max([",", ";", "\t"], key=header.count)
    if not header.count(sep):
        sep = ","
    body = "\n".join(text.splitlines()[1:20])
    decimal = "," if sep != "," and re.search(r"\d,\d", body) else "."
    return sep, decimal


def blank_template(columns: list[str], ageing_col: str, n_rows: int = 12) -> pd.DataFrame:
    """Empty history with the ageing column numbered 1..n so the table is ready to fill in."""
    df = pd.DataFrame(np.nan, index=range(n_rows), columns=columns, dtype=float)
    df[ageing_col] = np.arange(1, n_rows + 1, dtype=float)
    return df


def column_help(columns: list[str]) -> dict[str, str]:
    return {c: INPUT_DESCRIPTIONS.get(c, "not used by the models (ignored)") for c in columns}


def ignored_columns(history: pd.DataFrame, input_columns: list[str]) -> list[str]:
    return [c for c in history.columns if c not in input_columns]


def fingerprint(dataset: str, history: pd.DataFrame, rul_variant: str,
                assumptions: RangeAssumptions | None) -> str:
    """Identifies one set of inputs, to tell whether the shown result is out of date."""
    h = hashlib.sha256()
    h.update(f"{dataset}|{rul_variant}|{assumptions}|{list(history.columns)}".encode())
    h.update(pd.util.hash_pandas_object(history.astype(str), index=False).to_numpy().tobytes())
    return h.hexdigest()


def predict_or_error(predictor: VoltaraPredictor, history: pd.DataFrame,
                     assumptions: RangeAssumptions | None) -> tuple[pd.DataFrame | None, str | None]:
    """Run the inference module; turn input problems into a message instead of a crash."""
    try:
        return predictor.predict(history, range_assumptions=assumptions), None
    except ValueError as e:
        return None, f"Check the measurements: {e}"
    except Exception as e:  # noqa: BLE001 - shown to the user, never swallowed silently
        return None, f"Prediction failed ({type(e).__name__}: {e})"


# --------------------------------------------------------------------------- #
# End of life within a history
# --------------------------------------------------------------------------- #
def end_of_life_index(pred: pd.DataFrame) -> int | None:
    """Position of the first row at or past end of life (SoH <= 80 %), or None if never reached."""
    past = (pred.rul_note == EOL_NOTE).to_numpy()
    return int(past.argmax()) if past.any() else None


def default_cutoff(pred: pd.DataFrame) -> int:
    """Row to predict as of by default.

    NASA ran most cells well past 80 % SoH, so on the last row RUL is 0 by definition. A history that
    reaches end of life is shown as of the middle of its healthy life instead; otherwise the last row.
    """
    eol = end_of_life_index(pred)
    return len(pred) - 1 if eol is None or eol == 0 else (eol - 1) // 2


# --------------------------------------------------------------------------- #
# Summary of the latest row
# --------------------------------------------------------------------------- #
def latest_summary(pred: pd.DataFrame, predictor: VoltaraPredictor) -> dict:
    last = pred.iloc[-1]
    assessed = not str(last.anomaly_explanation).startswith("not assessed")
    out_of_range = pred.inputs_outside_training_range.iloc[-1]
    return {
        "ageing_col": predictor.order_col,
        "ageing": float(last[predictor.order_col]),
        "rows": len(pred),
        "soh_pred": float(last.soh_pred),
        "soh_measured": None if pd.isna(last.soh_measured) else float(last.soh_measured),
        "soh_source": last.soh_source,
        "rul": float(last.rul_pred),
        "rul_unit": last.rul_unit,
        "rul_note": last.rul_note,
        "rul_r2": float(predictor.rul_meta.get("loco_R2", np.nan)),
        "rul_mae": float(predictor.rul_meta.get("loco_MAE", np.nan)),
        "risk_rule": last.risk_rule,
        "risk_rul_model": last.risk_from_rul_model,
        "anomaly_assessed": assessed,
        "anomaly_latest": bool(last.anomaly_flag),
        "anomaly_explanation": last.anomaly_explanation,
        "anomaly_count": int(pred.anomaly_flag.sum()),
        "energy_retention": None if pd.isna(last.energy_retention) else float(last.energy_retention),
        "energy_source": last.energy_retention_source,
        "range_km": None if pd.isna(last.range_km_assumed) else float(last.range_km_assumed),
        "missing_inputs": [c for c in str(last.missing_inputs).split(", ") if c],
        "imputed_latest": [c for c in str(last.imputed_inputs).split(", ") if c],
        "out_of_range_latest": [c for c in str(out_of_range).split(", ") if c],
        "rows_out_of_range": int((pred.inputs_outside_training_range != "").sum()),
        "quality_flags_latest": last.data_quality_flags,
        "rows_quality_flags": int((pred.data_quality_flags != "").sum()),
        "status": last.prediction_status,
        "soh_coverage": float(last.soh_input_coverage),
        "rul_coverage": float(last.rul_input_coverage),
    }


def soh_domain(values: pd.Series) -> list[float]:
    """Always show 100 % and the EOL line, so a flat series is not magnified into a trend."""
    v = values.dropna()
    lo = min(float(v.min()) if len(v) else EOL_SOH, EOL_SOH - 0.05)
    hi = max(float(v.max()) if len(v) else 1.0, 1.0)
    return [lo - 0.02, hi + 0.02]


def risk_badge(level: str, prefix: str = "") -> str:
    """Status colour + icon + text: risk is never conveyed by colour alone."""
    color = RISK_COLORS.get(level, MUTED)
    return (f'<span class="v-pill" style="border:2px solid {color};box-shadow:0 0 0 4px {color}1f">'
            f'<span style="color:{color};font-size:1.1rem">{RISK_ICONS.get(level, "?")}</span>'
            f'{prefix}{level}</span>')


# --------------------------------------------------------------------------- #
# Charts (Altair; fixed palette order, one y-axis per chart, tooltips on marks)
# --------------------------------------------------------------------------- #
def _axis_x(col: str) -> alt.X:
    extra = {"tickMinStep": 1} if col == "cycle_number" else {}  # whole cycles only
    axis = alt.Axis(gridColor=GRID, format=",.0f", **extra)
    return alt.X(f"{col}:Q", title=AGEING_LABELS.get(col, col), axis=axis)


def soh_chart(pred: pd.DataFrame, ageing_col: str) -> alt.LayerChart:
    series = [("soh_pred", "Predicted SoH"), ("soh_measured", "Measured SoH")]  # measured drawn on top
    long = pd.concat([pred[[ageing_col]].assign(series=label, soh=pred[col])
                      for col, label in series if pred[col].notna().any()], ignore_index=True)
    domain = ["Measured SoH", "Predicted SoH", "Anomaly"]
    color = alt.Color("series:N", title=None, scale=alt.Scale(domain=domain,
                      range=[CATEGORICAL[0], CATEGORICAL[1], STATUS["critical"]]),
                      legend=alt.Legend(orient="top"))
    y = alt.Y("soh:Q", title="State of health", axis=alt.Axis(format=".0%", gridColor=GRID),
              scale=alt.Scale(domain=soh_domain(long.soh), clamp=True))
    tooltip = [alt.Tooltip(f"{ageing_col}:Q", title=AGEING_LABELS[ageing_col], format=".1f"),
               alt.Tooltip("series:N", title="Series"), alt.Tooltip("soh:Q", title="SoH", format=".1%")]
    dash = alt.StrokeDash("series:N", legend=None, scale=alt.Scale(
        domain=["Measured SoH", "Predicted SoH"], range=[[1, 0], [5, 3]]))
    lines = alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=22)).encode(
        x=_axis_x(ageing_col), y=y, color=color, strokeDash=dash, tooltip=tooltip,
        order=alt.Order("series:N", sort="descending"))
    ref = "soh_measured" if pred.soh_measured.notna().any() else "soh_pred"
    an = pred[pred.anomaly_flag].assign(series="Anomaly", soh=lambda d: d[ref])
    anomalies = alt.Chart(an).mark_point(size=140, filled=False, strokeWidth=2).encode(
        x=_axis_x(ageing_col), y=y, color=color,
        tooltip=[*tooltip, alt.Tooltip("anomaly_explanation:N", title="Why")])
    eol = alt.Chart(pd.DataFrame({"y": [EOL_SOH], "label": ["End of life (80 %)"]}))
    rule = eol.mark_rule(strokeDash=[4, 4], color=INK_2, strokeWidth=1).encode(y="y:Q")
    text = eol.mark_text(align="left", dx=4, dy=-6, color=INK_2, fontSize=11).encode(
        y="y:Q", x=alt.value(0), text="label:N")
    return (lines + anomalies + rule + text).properties(height=280)


def rul_chart(pred: pd.DataFrame, ageing_col: str) -> alt.LayerChart:
    unit = pred.rul_unit.iloc[0]
    base = alt.Chart(pred).encode(x=_axis_x(ageing_col),
                                  y=alt.Y("rul_pred:Q", title=f"Remaining useful life ({unit})",
                                          axis=alt.Axis(gridColor=GRID)))
    line = base.mark_line(color=MUTED, strokeWidth=1.5)
    pts = base.mark_point(filled=True, size=55).encode(
        color=alt.Color("risk_rule:N", title="Risk (rule)",
                        scale=alt.Scale(domain=LEVELS, range=[RISK_COLORS[k] for k in LEVELS]),
                        legend=alt.Legend(orient="top")),
        shape=alt.Shape("risk_rule:N", title="Risk (rule)",
                        scale=alt.Scale(domain=LEVELS, range=["circle", "triangle-up", "square"])),
        tooltip=[alt.Tooltip(f"{ageing_col}:Q", title=AGEING_LABELS[ageing_col], format=".1f"),
                 alt.Tooltip("rul_pred:Q", title=f"RUL ({unit})", format=".0f"),
                 alt.Tooltip("risk_rule:N", title="Risk (rule)"),
                 alt.Tooltip("risk_from_rul_model:N", title="Risk (RUL model)")])
    return (line + pts).properties(height=280)


def energy_chart(pred: pd.DataFrame, ageing_col: str) -> alt.Chart:
    return alt.Chart(pred).mark_line(color=CATEGORICAL[2], strokeWidth=2,
                                     point=alt.OverlayMarkDef(size=28, color=CATEGORICAL[2])).encode(
        x=_axis_x(ageing_col),
        y=alt.Y("energy_retention:Q", title="Energy retention (vs. first rows)",
                axis=alt.Axis(format=".0%", gridColor=GRID),
                scale=alt.Scale(domain=soh_domain(pred.energy_retention), clamp=True)),
        tooltip=[alt.Tooltip(f"{ageing_col}:Q", title=AGEING_LABELS[ageing_col], format=".1f"),
                 alt.Tooltip("energy_retention:Q", title="Energy retention", format=".1%")],
    ).properties(height=280)


def ev_soh_chart(history: pd.DataFrame) -> alt.LayerChart:
    """EV page: predicted (and, for dataset samples, recorded) SoH in % against age in years."""
    series = [("soh_pred", "Predicted SoH"), ("soh_measured", "Recorded SoH (dataset label)")]
    long = pd.concat([history[["age_years"]].assign(series=label, soh=history[col])
                      for col, label in series if col in history and history[col].notna().any()],
                     ignore_index=True).dropna()
    lo = float(min(long.soh.min(), EOL_SOH * 100)) - 2
    color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=[s for _, s in series], range=[CATEGORICAL[1], CATEGORICAL[0]]))
    lines = alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=22)).encode(
        x=alt.X("age_years:Q", title="Battery age (years)", axis=alt.Axis(gridColor=GRID)),
        y=alt.Y("soh:Q", title="State of health (%)", scale=alt.Scale(domain=[lo, 101]),
                axis=alt.Axis(gridColor=GRID)),
        color=color, tooltip=[alt.Tooltip("age_years:Q", title="Age (years)", format=".2f"),
                              alt.Tooltip("series:N", title="Series"), alt.Tooltip("soh:Q", title="SoH %", format=".1f")])
    eol = alt.Chart(pd.DataFrame({"y": [EOL_SOH * 100]})).mark_rule(strokeDash=[4, 4], color=INK_2).encode(y="y:Q")
    return (lines + eol).properties(height=260)
