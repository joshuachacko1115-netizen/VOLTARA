"""Inference: load the saved VOLTARA models and score one battery's measurement history.

Input is the history of ONE battery, one row per discharge cycle (Battery Aging
format) or per reference check (Randomized Usage format), in ageing order, with
the same feature columns the models were trained on (see the model .json files,
or `VoltaraPredictor.required_inputs()`). Missing columns are allowed: they are
imputed with the training median and reported in `missing_inputs`.

Optional measured columns that improve the output:
  capacity_ah   measured capacity per row (a capacity test). If present, RUL and
                risk use the measured SoH; otherwise they use the predicted SoH
                and this is stated in `soh_source`.
  energy_wh     measured discharge energy per row. If present, energy retention is
                *measured*; otherwise a capacity-based proxy is returned and labelled.

For Battery Aging, raw charge/discharge signals can be turned into a feature row
with `battery_aging_features_from_signals()`.

Outputs per row: predicted SoH, RUL (with unit and caveats), risk class (rule-based
and RUL-model-based), anomaly flags with explanations, input range warnings, and
energy retention / range where range in km is only produced from an explicit
RangeAssumptions object and is always labelled as assumption-based.

Usage:
    python -m src.inference.predict --dataset battery_aging --cell B0005
    python -m src.inference.predict --dataset randomized_usage --csv my_checks.csv
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import BA_RATED_CAPACITY_AH, EOL_SOH, MODELS_DIR, PROCESSED_DIR
from src.features.battery_aging import charge_features, discharge_features
from src.models.anomaly import robust_z_within_cell
from src.models.range_estimation import RangeAssumptions, estimate_range_km
from src.models.risk import classes_from_rul, classify
from src.preprocessing.cleaning import causal_initial, causal_rolling, causal_rolling_slope, residual_vs_recent

TAGS = {"battery_aging": "ba", "randomized_usage": "rw"}
MISSING_TOKENS = {"", "-", "--", "?", "na", "n/a", "nan", "none", "null", "missing"}
MAX_ABS_VALUE = 1e9  # no battery measurement here comes near this


def _norm(name) -> str:
    return "".join(ch for ch in str(name).strip().lower().replace("-", "_").replace(" ", "_") if ch.isprintable())


def _column_matcher(known: list[str], given) -> dict:
    """Rename given columns that match a known input ignoring case, spaces and hyphens."""
    lookup = {_norm(k): k for k in known}
    return {g: lookup[_norm(g)] for g in given if g not in known and _norm(g) in lookup
            and lookup[_norm(g)] not in given}
OPTIONAL_MEASUREMENTS = ["capacity_ah", "energy_wh"]
INPUT_DESCRIPTIONS = {
    # shared
    "elapsed_days": "days since the first measurement",
    "capacity_ah": "measured capacity, Ah (optional: a capacity test)",
    "energy_wh": "measured discharge energy, Wh (optional)",
    # Battery Aging: one row per discharge cycle
    "cycle_number": "discharge cycle number (ageing axis)",
    "ambient_temperature": "ambient temperature, °C",
    "i_set": "discharge current setpoint, A",
    "chg_ah": "charge put in by the preceding charge, Ah",
    "chg_cc_duration_s": "constant-current charge time, s",
    "chg_cv_duration_s": "constant-voltage charge time, s",
    "chg_t_max": "max temperature during charge, °C",
    "chg_t_mean": "mean temperature during charge, °C",
    "chg_v_start": "voltage at the start of charge, V",
    "dis_t_max": "max temperature during discharge, °C",
    "re_ohm": "electrolyte resistance (impedance test), Ω",
    "rct_ohm": "charge-transfer resistance (impedance test), Ω",
    "impedance_age_cycles": "cycles since the last impedance test",
    "rest_before_h": "rest before this cycle, h",
    "v_at_60s": "voltage 60 s into the discharge, V (anomaly detection)",
    "v_at_300s": "voltage 300 s into the discharge, V (anomaly detection)",
    "dis_t_start": "temperature at the start of discharge, °C (anomaly detection)",
    "t_rise_300s": "temperature rise over the first 300 s of discharge, °C (anomaly detection)",
    "dis_v_std": "standard deviation of the discharge voltage, V (anomaly detection)",
    # Randomized Usage: one row per reference check
    "efc": "equivalent full cycles = discharged Ah / initial capacity (ageing axis)",
    "interval_discharge_ah": "Ah discharged since the previous check",
    "interval_rw_i_mean": "mean load current since the previous check, A",
    "interval_rw_t_mean": "mean temperature since the previous check, °C",
    "interval_rw_t_max": "max temperature since the previous check, °C",
    "interval_frac_hit_vmin": "share of load steps that reached the minimum voltage",
    "r_operational_ohm": "resistance from load steps, Ω",
    "r_pulse_ohm": "resistance from the pulse test, Ω",
    "ref_chg_cc_duration_s": "constant-current time of the 2 A reference charge, s",
    "ref_chg_cv_duration_s": "constant-voltage time of the 2 A reference charge, s",
    "ref_chg_t_max": "max temperature during the reference charge, °C",
    "ref_t_max": "max temperature during the reference discharge, °C (anomaly detection)",
    "ambient_40c": "1 if the ambient is 40 °C, else 0",
    "dist_skewed_high": "1 if the load profile is skewed to high currents",
    "dist_skewed_low": "1 if the load profile is skewed to low currents",
    "dist_uniform": "1 if the load profile is uniform",
}
MIN_HISTORY_FOR_ANOMALY = 10
# Below this share of measured (not imputed) model inputs, a row's output mostly reflects the
# training medians used for imputation, so it is reported as "insufficient inputs".
MIN_INPUT_COVERAGE = 0.5
RUL_VARIANTS = {"default": "", "censoring_aware": "_censoring_aware"}
EOL_NOTE = "at or past end of life (SoH <= 80 %)"
CENSORING_NOTE = ("RUL models were trained only on cells that reached 80 % SoH; slow-fading "
                  "batteries are under-represented, so RUL may be underestimated for them.")
CENSORING_AWARE_NOTE = ("RUL model also trained on never-failed cells (imputed lives); fewer "
                        "underestimates for slow-fading batteries, slightly larger error on cells that failed.")


def battery_aging_features_from_signals(discharge: pd.DataFrame, charge: pd.DataFrame | None = None,
                                        **context) -> dict:
    """Build one Battery Aging feature row from raw signals.

    discharge / charge: DataFrames with columns time_s, voltage_measured,
    current_measured (A, negative = discharge, as in the NASA files) and
    temperature_measured. `context` adds known values such as cycle_number,
    ambient_temperature, elapsed_days, rest_before_h, re_ohm, rct_ohm.
    """
    row = dict(context)
    row.update(discharge_features(discharge))
    if charge is not None:
        row.update(charge_features(charge))
    if "capacity_ah_2v7" in row:
        row.setdefault("capacity_ah", row["capacity_ah_2v7"])
    return row


class VoltaraPredictor:
    def __init__(self, dataset: str = "battery_aging", models_dir: Path = MODELS_DIR,
                 rul_variant: str = "default"):
        """rul_variant: "default" (trained on cells that reached EOL) or "censoring_aware"
        (also uses never-failed cells; see src/models/censored_rul.py for the trade-off)."""
        if dataset not in TAGS:
            raise ValueError(f"dataset must be one of {list(TAGS)}")
        self.dataset, self.tag, d = dataset, TAGS[dataset], Path(models_dir)
        self.soh_model = joblib.load(d / f"soh_{self.tag}.joblib")
        self.soh_meta = json.loads((d / f"soh_{self.tag}.json").read_text())
        if rul_variant not in RUL_VARIANTS:
            raise ValueError(f"rul_variant must be one of {list(RUL_VARIANTS)}")
        rul_file = f"rul_{self.tag}{RUL_VARIANTS[rul_variant]}"
        self.rul_variant = rul_variant
        self.rul_model = joblib.load(d / f"{rul_file}.joblib")
        self.rul_meta = json.loads((d / f"{rul_file}.json").read_text())
        self.risk_meta = json.loads((d / f"risk_{self.tag}.json").read_text())
        self.anomaly_model = joblib.load(d / f"anomaly_{self.tag}.joblib")
        self.anomaly_meta = json.loads((d / f"anomaly_{self.tag}.json").read_text())
        self.order_col = "cycle_number" if self.tag == "ba" else "efc"

    # ------------------------------------------------------------------ #
    @staticmethod
    def input_columns(dataset: str, models_dir: Path = MODELS_DIR) -> list[str]:
        """Ageing column first, then the model inputs, then the optional measured columns."""
        p = VoltaraPredictor(dataset, models_dir)
        rest = [c for c in p.required_inputs() if c != p.order_col]
        return [p.order_col, *rest, *OPTIONAL_MEASUREMENTS]

    def required_inputs(self) -> list[str]:
        """Measured columns the models use: SoH and RUL inputs, plus the anomaly detector's raw
        inputs (its capacity-based features are derived from capacity_ah)."""
        trend = set(self._trend_columns())
        cols = set(self.soh_meta["features"]) | (set(self.rul_meta["features"]) - trend)
        cols |= set(self.anomaly_meta["features"]) - set(self._capacity_columns())
        return sorted(cols | {self.order_col})

    def _capacity_columns(self) -> list[str]:
        """Anomaly features derived from measured capacity, as in src/features/targets.py."""
        return ["capacity_ah_2v7", "cap_resid_prev5"] if self.tag == "ba" else ["soh", "cap_resid_prev3"]

    def _add_capacity_features(self, df: pd.DataFrame, soh: pd.Series, source: str) -> pd.DataFrame:
        if source != "measured":  # without a capacity test there is nothing measured to compare
            for c in self._capacity_columns():
                df[c] = np.nan
        elif self.tag == "ba":
            df["capacity_ah_2v7"] = df.capacity_ah
            df["cap_resid_prev5"] = residual_vs_recent(df.capacity_ah, 5)
        else:
            df["soh"] = soh
            df["cap_resid_prev3"] = residual_vs_recent(soh, 3)
        return df

    def _trend_columns(self) -> list[str]:
        if self.tag == "ba":
            return ["soh_roll_med5", "cap_roll_std5", "fade_slope_10", "fade_slope_20", "capacity_fade_ah"]
        return ["soh_roll_med3", "fade_slope_4", "fade_slope_8"]

    def _soh_for_trends(self, df: pd.DataFrame) -> tuple[pd.Series, str]:
        if "capacity_ah" in df and df.capacity_ah.notna().any():
            if self.tag == "ba":
                return df.capacity_ah / BA_RATED_CAPACITY_AH, "measured"
            first = df.capacity_ah.dropna().iloc[0]
            return df.capacity_ah / first, "measured"
        return df.soh_pred, "predicted"

    def _add_trends(self, df: pd.DataFrame, soh: pd.Series) -> pd.DataFrame:
        """Same causal trend features as src/features/targets.py, from the SoH series."""
        x = df[self.order_col].to_numpy(float)
        s = soh.to_numpy(float)
        if self.tag == "ba":
            cap = s * BA_RATED_CAPACITY_AH
            med5 = causal_rolling(cap, 5, "median")
            df["soh_roll_med5"] = med5 / BA_RATED_CAPACITY_AH
            df["cap_roll_std5"] = causal_rolling(cap, 5, "std")
            df["fade_slope_10"] = 100 * causal_rolling_slope(cap, x, window=10)
            df["fade_slope_20"] = 100 * causal_rolling_slope(cap, x, window=20)
            df["capacity_fade_ah"] = causal_initial(cap, 3) - med5
        else:
            df["soh_roll_med3"] = causal_rolling(s, 3, "median")
            df["fade_slope_4"] = 100 * causal_rolling_slope(s, x, window=4)
            df["fade_slope_8"] = 100 * causal_rolling_slope(s, x, window=8)
        return df

    def _range_warnings(self, df: pd.DataFrame, meta: dict) -> pd.Series:
        ranges = meta.get("feature_ranges", {})
        out = []
        for _, r in df.iterrows():
            bad = [f for f, (lo, hi) in ranges.items()
                   if f in r and pd.notna(r[f]) and not (lo <= r[f] <= hi)]
            out.append(", ".join(bad))
        return pd.Series(out, index=df.index)

    def _anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = self.anomaly_meta["features"]
        res = pd.DataFrame(index=df.index)
        res["anomaly_flag"], res["anomaly_score"], res["anomaly_explanation"] = False, np.nan, ""
        frame = df.reindex(columns=feats).assign(cell="this_battery")
        if len(df) < MIN_HISTORY_FOR_ANOMALY:
            res["anomaly_explanation"] = (f"not assessed: needs >= {MIN_HISTORY_FOR_ANOMALY} rows of "
                                          "history to know what is normal for this battery")
            return res
        z = robust_z_within_cell(frame, feats).fillna(0.0)
        res["anomaly_score"] = -self.anomaly_model.score_samples(z)
        res["anomaly_flag"] = self.anomaly_model.predict(z) == -1
        res["anomaly_explanation"] = [
            "; ".join(f"{f} {z.loc[i, f]:+.1f}σ from this battery's typical value"
                      for f in z.loc[i].abs().nlargest(3).index) if res.anomaly_flag[i] else ""
            for i in z.index]
        return res

    def _quality_flags(self, df: pd.DataFrame) -> pd.Series:
        """Rule-based data-quality checks (Battery Aging style columns, when present)."""
        notes = pd.Series("", index=df.index)
        if "i_set" in df:
            notes = notes.where(~(df.i_set.fillna(1) < 0.5), notes + "no load current; ")
        if "reached_cutoff" in df:
            notes = notes.where(df.reached_cutoff.fillna(True).astype(bool),
                                notes + "discharge did not reach the cutoff voltage; ")
        if "capacity_ah" in df:
            notes = notes.where(~(df.capacity_ah < 0.3), notes + "implausibly low capacity; ")
        if "energy_wh" in df:
            notes = notes.where(~(df.energy_wh <= 0), notes + "energy not positive; ")
        dup = df[self.order_col].duplicated(keep=False)
        notes = notes.where(~dup, notes + f"same {self.order_col} as another row; ")
        return notes.str.rstrip("; ")

    def validate(self, history: pd.DataFrame) -> pd.DataFrame:
        """Check and normalise an input history; raise ValueError with a readable message."""
        if not isinstance(history, pd.DataFrame) or history.empty:
            raise ValueError("the measurement history is empty: give at least one row")
        df = history.copy().reset_index(drop=True)
        known = [*self.required_inputs(), *OPTIONAL_MEASUREMENTS]
        df = df.rename(columns=_column_matcher(known, df.columns))
        if self.order_col not in df:
            other = next((d for d, tag in TAGS.items() if tag != self.tag), None)
            other_col = "efc" if self.tag == "ba" else "cycle_number"
            hint = (f" This looks like {other.replace('_', ' ').title()} data (it has '{other_col}'): "
                    "switch the battery dataset." if other_col in df else "")
            raise ValueError(f"the history needs an ageing column '{self.order_col}' "
                             f"({INPUT_DESCRIPTIONS[self.order_col]}).{hint}")
        numeric = [c for c in known if c in df]
        bad, nonfinite = {}, {}
        for c in numeric:
            col = df[c]
            if col.dtype == object or pd.api.types.is_string_dtype(col):
                col = col.astype("string").str.strip()
                col = col.mask(col.str.lower().isin(MISSING_TOKENS))
            conv = pd.to_numeric(col, errors="coerce")
            wrong = conv.isna() & col.notna()
            if wrong.any():
                bad[c] = col[wrong].astype(str).unique()[:3].tolist()
            conv = conv.astype(float)
            # Infinite or absurdly large values break the models (float32 trees) and are never real.
            huge = conv.notna() & ~(conv.abs() < MAX_ABS_VALUE)
            if huge.any():
                nonfinite[c] = conv[huge].unique()[:3].tolist()
            df[c] = conv
        if bad:
            raise ValueError("non-numeric values in " + "; ".join(
                f"'{c}' (e.g. {', '.join(repr(x) for x in v)})" for c, v in bad.items()))
        if nonfinite:
            raise ValueError("infinite or implausibly large values in " + "; ".join(
                f"'{c}' (e.g. {', '.join(f'{x:g}' for x in v)})" for c, v in nonfinite.items()))
        if df[self.order_col].isna().any():
            raise ValueError(f"'{self.order_col}' must be filled in on every row "
                             f"({df[self.order_col].isna().sum()} empty)")
        if (df[self.order_col] < 0).any():
            raise ValueError(f"'{self.order_col}' cannot be negative")
        return df.sort_values(self.order_col, kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    def predict(self, history: pd.DataFrame, range_assumptions: RangeAssumptions | None = None
                ) -> pd.DataFrame:
        df = self.validate(history)
        missing = [c for c in self.required_inputs() if c not in df or df[c].isna().all()]
        for c in missing:
            df[c] = np.nan
        row_gaps = df[self.required_inputs()].isna()

        out = pd.DataFrame({self.order_col: df[self.order_col]})
        df["soh_pred"] = np.clip(self.soh_model.predict(df[self.soh_meta["features"]]), 0, 1.2)
        out["soh_pred"] = df.soh_pred
        soh_trend, source = self._soh_for_trends(df)
        out["soh_measured"] = soh_trend if source == "measured" else np.nan
        out["soh_source"] = source
        df = self._add_trends(df, soh_trend)
        df = self._add_capacity_features(df, soh_trend, source)

        # RUL
        rul = np.clip(self.rul_model.predict(df[self.rul_meta["features"]]), 0, None)
        soh_now = df["soh_roll_med5" if self.tag == "ba" else "soh_roll_med3"]
        rul = np.where(soh_now <= EOL_SOH, 0.0, rul)
        out["rul_pred"], out["rul_unit"] = rul, self.rul_meta["unit"]
        note = CENSORING_NOTE if self.rul_variant == "default" else CENSORING_AWARE_NOTE
        out["rul_note"] = np.where(soh_now <= EOL_SOH, EOL_NOTE, note)

        # Risk
        rm = self.risk_meta
        rate = -df[rm["slope_col"]] * rm["slope_to_soh_per_unit"]
        projected = ((soh_now - EOL_SOH) / rate).where(rate > 0, np.inf)
        projected = projected.where(soh_now > EOL_SOH, 0.0).fillna(np.inf)
        out["risk_rule"] = classify(soh_now, projected, rm["t1"], rm["t2"]).to_numpy()
        out["risk_from_rul_model"] = classes_from_rul(rul, rm["r1"], rm["r2"])

        # Anomalies and input checks
        out = out.join(self._anomalies(df))
        out["data_quality_flags"] = self._quality_flags(df)
        out["inputs_outside_training_range"] = self._range_warnings(
            df, {"feature_ranges": {**self.soh_meta.get("feature_ranges", {}),
                                    **self.rul_meta.get("feature_ranges", {})}})
        out["missing_inputs"] = ", ".join(missing)
        out["soh_input_coverage"] = 1 - row_gaps[self.soh_meta["features"]].mean(axis=1)
        rul_raw = [c for c in self.rul_meta["features"] if c in row_gaps]  # trend features are derived
        out["rul_input_coverage"] = 1 - row_gaps[rul_raw].mean(axis=1)
        low = np.minimum(out.soh_input_coverage, out.rul_input_coverage)
        out["prediction_status"] = np.select(
            [low < MIN_INPUT_COVERAGE, low < 1], ["insufficient inputs", "partial inputs"], "ok")
        out["imputed_inputs"] = [", ".join(c for c in row_gaps.columns[r] if c not in missing)
                                 for r in row_gaps.to_numpy()]

        # Energy retention (measured if possible) and assumption-based range
        if "energy_wh" in df and df.energy_wh.notna().sum() >= 1:
            e = df.energy_wh
            out["energy_retention"] = e / causal_initial(e.ffill().bfill(), 3)
            out["energy_retention_source"] = "measured (discharge energy)"
            cell_energy = e
        else:
            out["energy_retention"] = soh_trend / causal_initial(soh_trend.ffill().bfill(), 3)
            out["energy_retention_source"] = ("proxy: capacity retention (energy not measured)"
                                              if source == "measured" else
                                              "proxy: predicted SoH (neither energy nor capacity measured)")
            cell_energy = None
        if range_assumptions is not None and cell_energy is not None:
            out["range_km_assumed"] = estimate_range_km(cell_energy, range_assumptions)
            out.attrs["range_assumptions"] = asdict(range_assumptions)
        else:
            out["range_km_assumed"] = np.nan
        out.attrs["notes"] = {
            "range_km_assumed": "only computed from measured energy AND explicit RangeAssumptions; "
                                "not validated against any vehicle",
            "anomaly_model": "Isolation Forest on z-scores within this history; capacity features "
                             + ("used (capacity_ah given)" if source == "measured"
                                else "NOT used (no capacity_ah), so capacity drops cannot be flagged"),
            "soh_model": f"{self.soh_meta['model']} ({self.soh_meta['feature_set']}), "
                         f"leave-one-cell-out MAE {self.soh_meta['loco_MAE']:.3f}",
            "rul_model": f"{self.rul_meta['model']}, leave-one-cell-out MAE "
                         f"{self.rul_meta['loco_MAE']:.1f} {self.rul_meta['unit']}, "
                         f"R2 {self.rul_meta['loco_R2']:.2f}"
                         + (" (below 0: not better than the training mean on unseen cells; "
                            "treat RUL as indicative only)" if self.rul_meta["loco_R2"] < 0 else ""),
        }
        return out

    def summary(self, history: pd.DataFrame, **kw) -> dict:
        """Prediction for the most recent row, as a plain dict."""
        p = self.predict(history, **kw)
        last = p.iloc[-1].to_dict()
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in last.items()} | {
            "notes": p.attrs.get("notes", {}), "range_assumptions": p.attrs.get("range_assumptions")}


def example_cells(dataset: str) -> list[str]:
    """NASA cells available as examples (needs the processed data from the pipeline)."""
    if dataset == "battery_aging":
        df = pd.read_parquet(PROCESSED_DIR / "ba_cycles.parquet", columns=["cell", "is_valid", "in_model_cohort"])
        cells = df[df.is_valid & df.in_model_cohort].cell.unique()
    else:
        df = pd.read_parquet(PROCESSED_DIR / "rw_checks.parquet", columns=["cell", "is_valid"])
        cells = df[df.is_valid].cell.unique()
    return sorted(cells, key=lambda c: int("".join(ch for ch in c if ch.isdigit())))


def example_history(dataset: str, cell: str, inputs_only: bool = True) -> pd.DataFrame:
    """Measurement history of one NASA cell in the predictor's input format."""
    hist = _demo_history(dataset, cell)
    if inputs_only:
        p_cols = VoltaraPredictor.input_columns(dataset)
        hist = hist[[c for c in p_cols if c in hist]]
    return hist.reset_index(drop=True)


def _demo_history(dataset: str, cell: str) -> pd.DataFrame:
    if dataset == "battery_aging":
        df = pd.read_parquet(PROCESSED_DIR / "ba_cycles.parquet")
        df = df[(df.cell == cell) & df.is_valid].rename(columns={"capacity_ah_2v7": "capacity_ah"})
    else:
        df = pd.read_parquet(PROCESSED_DIR / "rw_checks.parquet")
        df = df[(df.cell == cell) & df.is_valid].rename(columns={"ref_energy_wh": "energy_wh"})
    if df.empty:
        raise ValueError(f"no valid rows for {cell}")
    # Keep only raw measurements: drop targets and pre-computed trend features.
    drop = [c for c in df.columns if c.startswith(("rul", "eol", "soh", "fade_slope", "cap_roll",
                                                    "capacity_fade", "cap_resid"))]
    return df.drop(columns=drop)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="battery_aging", choices=list(TAGS))
    ap.add_argument("--cell", help="score a cell from the processed NASA data (demo)")
    ap.add_argument("--csv", help="score your own history CSV (one battery, one row per cycle/check)")
    ap.add_argument("--rows", type=int, default=5, help="rows to print")
    ap.add_argument("--rul-variant", default="default", choices=list(RUL_VARIANTS))
    args = ap.parse_args()
    if not (args.cell or args.csv):
        ap.error("give --cell or --csv")
    hist = (pd.read_csv(args.csv, float_precision="round_trip") if args.csv
            else _demo_history(args.dataset, args.cell))
    pred = VoltaraPredictor(args.dataset, rul_variant=args.rul_variant).predict(hist)
    pd.set_option("display.width", 220, "display.max_colwidth", 60)
    cols = [c for c in pred.columns if c not in ("rul_note", "missing_inputs")]
    print(pred[cols].tail(args.rows).round(3).to_string(index=False))
    print("\nNotes:", json.dumps(pred.attrs["notes"], indent=2))
    if pred.missing_inputs.iloc[0]:
        print("Missing inputs (imputed with the training median):", pred.missing_inputs.iloc[0])


if __name__ == "__main__":
    main()
