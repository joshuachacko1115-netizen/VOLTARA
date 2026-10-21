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
from src.preprocessing.cleaning import causal_initial, causal_rolling, causal_rolling_slope

TAGS = {"battery_aging": "ba", "randomized_usage": "rw"}
MIN_HISTORY_FOR_ANOMALY = 10
RUL_VARIANTS = {"default": "", "censoring_aware": "_censoring_aware"}
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
    def required_inputs(self) -> list[str]:
        trend = set(self._trend_columns())
        cols = set(self.soh_meta["features"]) | (set(self.rul_meta["features"]) - trend)
        return sorted(cols | {self.order_col})

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
        return notes.str.rstrip("; ")

    # ------------------------------------------------------------------ #
    def predict(self, history: pd.DataFrame, range_assumptions: RangeAssumptions | None = None
                ) -> pd.DataFrame:
        df = history.copy().reset_index(drop=True)
        if self.order_col not in df:
            raise ValueError(f"history needs an ageing column '{self.order_col}'")
        df = df.sort_values(self.order_col).reset_index(drop=True)
        missing = [c for c in self.required_inputs() if c not in df]
        for c in missing:
            df[c] = np.nan

        out = pd.DataFrame({self.order_col: df[self.order_col]})
        df["soh_pred"] = np.clip(self.soh_model.predict(df[self.soh_meta["features"]]), 0, 1.2)
        out["soh_pred"] = df.soh_pred
        soh_trend, source = self._soh_for_trends(df)
        out["soh_measured"] = soh_trend if source == "measured" else np.nan
        out["soh_source"] = source
        df = self._add_trends(df, soh_trend)

        # RUL
        rul = np.clip(self.rul_model.predict(df[self.rul_meta["features"]]), 0, None)
        soh_now = df["soh_roll_med5" if self.tag == "ba" else "soh_roll_med3"]
        rul = np.where(soh_now <= EOL_SOH, 0.0, rul)
        out["rul_pred"], out["rul_unit"] = rul, self.rul_meta["unit"]
        note = CENSORING_NOTE if self.rul_variant == "default" else CENSORING_AWARE_NOTE
        out["rul_note"] = np.where(soh_now <= EOL_SOH, "at or past end of life (SoH <= 80 %)", note)

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

        # Energy retention (measured if possible) and assumption-based range
        if "energy_wh" in df and df.energy_wh.notna().sum() >= 1:
            e = df.energy_wh
            out["energy_retention"] = e / causal_initial(e.ffill().bfill(), 3)
            out["energy_retention_source"] = "measured (discharge energy)"
            cell_energy = e
        else:
            out["energy_retention"] = soh_trend / causal_initial(soh_trend.ffill().bfill(), 3)
            out["energy_retention_source"] = "proxy: capacity retention (energy not measured)"
            cell_energy = None
        if range_assumptions is not None and cell_energy is not None:
            out["range_km_assumed"] = estimate_range_km(cell_energy, range_assumptions)
            out.attrs["range_assumptions"] = asdict(range_assumptions)
        else:
            out["range_km_assumed"] = np.nan
        out.attrs["notes"] = {
            "range_km_assumed": "only computed from measured energy AND explicit RangeAssumptions; "
                                "not validated against any vehicle",
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


def _demo_history(dataset: str, cell: str) -> pd.DataFrame:
    if dataset == "battery_aging":
        df = pd.read_parquet(PROCESSED_DIR / "ba_cycles.parquet")
        df = df[(df.cell == cell) & df.is_valid].rename(columns={"capacity_ah_2v7": "capacity_ah"})
    else:
        df = pd.read_parquet(PROCESSED_DIR / "rw_checks.parquet")
        df = df[(df.cell == cell) & df.is_valid].rename(columns={"ref_energy_wh": "energy_wh"})
    if df.empty:
        raise SystemExit(f"no valid rows for {cell}")
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
    hist = pd.read_csv(args.csv) if args.csv else _demo_history(args.dataset, args.cell)
    pred = VoltaraPredictor(args.dataset, rul_variant=args.rul_variant).predict(hist)
    pd.set_option("display.width", 220, "display.max_colwidth", 60)
    cols = [c for c in pred.columns if c not in ("rul_note", "missing_inputs")]
    print(pred[cols].tail(args.rows).round(3).to_string(index=False))
    print("\nNotes:", json.dumps(pred.attrs["notes"], indent=2))
    if pred.missing_inputs.iloc[0]:
        print("Missing inputs (imputed with the training median):", pred.missing_inputs.iloc[0])


if __name__ == "__main__":
    main()
