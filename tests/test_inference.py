"""Inference-module tests. They need the trained artifacts in models/ (run the pipeline first)."""
import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR, PROCESSED_DIR
from src.inference.predict import VoltaraPredictor, _demo_history
from src.models.range_estimation import RangeAssumptions

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "anomaly_ba.joblib").exists() or not (PROCESSED_DIR / "ba_cycles.parquet").exists(),
    reason="trained models / processed data not available - run `python -m src.run_pipeline`")


@pytest.mark.parametrize("dataset,cell", [("battery_aging", "B0005"), ("randomized_usage", "RW9")])
def test_predict_returns_all_outputs(dataset, cell):
    pred = VoltaraPredictor(dataset).predict(_demo_history(dataset, cell))
    for col in ["soh_pred", "rul_pred", "rul_unit", "risk_rule", "risk_from_rul_model", "anomaly_flag",
                "energy_retention", "energy_retention_source", "range_km_assumed"]:
        assert col in pred
    assert pred.soh_pred.between(0, 1.2).all()
    assert (pred.rul_pred >= 0).all()
    assert set(pred.risk_rule) <= {"Low", "Medium", "High"}
    assert pred.soh_source.iloc[0] == "measured"
    assert pred.energy_retention_source.iloc[0].startswith("measured")


def test_range_in_km_requires_explicit_assumptions():
    hist = _demo_history("battery_aging", "B0005")
    p = VoltaraPredictor("battery_aging")
    assert p.predict(hist).range_km_assumed.isna().all()
    with_assumptions = p.predict(hist, range_assumptions=RangeAssumptions())
    assert with_assumptions.range_km_assumed.notna().all()
    assert "range_assumptions" in with_assumptions.attrs


def test_without_capacity_uses_predicted_soh_and_says_so():
    hist = _demo_history("battery_aging", "B0006").drop(columns=["capacity_ah", "energy_wh"])
    pred = VoltaraPredictor("battery_aging").predict(hist)
    assert (pred.soh_source == "predicted").all()
    assert pred.energy_retention_source.iloc[0].startswith("proxy")


def test_short_history_skips_anomaly_detection_and_missing_columns_are_reported():
    hist = _demo_history("battery_aging", "B0005").head(4)[["cycle_number", "chg_cc_duration_s"]]
    pred = VoltaraPredictor("battery_aging").predict(hist)
    assert not pred.anomaly_flag.any()
    assert pred.anomaly_explanation.str.startswith("not assessed").all()
    assert "re_ohm" in pred.missing_inputs.iloc[0]


def test_trend_features_are_causal():
    hist = _demo_history("battery_aging", "B0007")
    p = VoltaraPredictor("battery_aging")
    full = p.predict(hist)
    part = p.predict(hist.head(60))
    # Predictions for the first 60 cycles must not change when later cycles are added.
    np.testing.assert_allclose(full.rul_pred.iloc[:60].to_numpy(), part.rul_pred.to_numpy())
    assert (full.risk_rule.iloc[:60].to_numpy() == part.risk_rule.to_numpy()).all()


@pytest.mark.skipif(not (MODELS_DIR / "rul_rw_censoring_aware.joblib").exists(),
                    reason="run `python -m src.models.censored_rul` first")
def test_censoring_aware_rul_variant_loads_and_says_so():
    hist = _demo_history("randomized_usage", "RW27")
    default = VoltaraPredictor("randomized_usage").predict(hist)
    aware = VoltaraPredictor("randomized_usage", rul_variant="censoring_aware").predict(hist)
    assert (aware.rul_pred >= 0).all()
    assert aware.rul_note.iloc[0] != default.rul_note.iloc[0]
    with pytest.raises(ValueError):
        VoltaraPredictor("randomized_usage", rul_variant="nope")


@pytest.mark.parametrize("dataset,cells", [("battery_aging", ["B0018", "B0036"]),
                                           ("randomized_usage", ["RW9", "RW11"])])
def test_anomaly_flags_match_the_pipeline_on_valid_rows(dataset, cells):
    """Same forest, same features: inference must flag exactly the pipeline's (valid-row) anomalies."""
    import pandas as pd
    from src.config import REPORTS_DIR
    from src.inference.predict import example_history
    tag = "ba" if dataset == "battery_aging" else "rw"
    ref = pd.read_csv(REPORTS_DIR / "results" / f"anomalies_{tag}.csv")
    data = pd.read_parquet(PROCESSED_DIR / ("ba_cycles.parquet" if tag == "ba" else "rw_checks.parquet"))
    key = "cycle_number" if tag == "ba" else "check_id"
    ref = ref.merge(data[["cell", key, "is_valid"]], on=["cell", key])
    p = VoltaraPredictor(dataset)
    for cell in cells:
        pred = p.predict(example_history(dataset, cell))
        order = p.order_col
        expected = ref[(ref.cell == cell) & ref.is_valid]
        exp_vals = set(expected[order].round(6)) if order in expected else set(
            data[(data.cell == cell) & data[key].isin(expected[key])][order].round(6))
        assert set(pred[pred.anomaly_flag][order].round(6)) == exp_vals, cell


def test_input_coverage_and_status():
    p = VoltaraPredictor("battery_aging")
    hist = _demo_history("battery_aging", "B0005")
    full = p.predict(hist)
    # early cycles precede the first impedance test, so they are "partial"; later rows are complete
    assert full.prediction_status.iloc[-1] == "ok" and full.soh_input_coverage.iloc[-1] == 1
    assert (full.prediction_status != "insufficient inputs").all()
    blank = hist[["cycle_number"]].copy()
    out = p.predict(blank)
    assert (out.prediction_status == "insufficient inputs").all()
    assert out.energy_retention_source.iloc[0].startswith("proxy: predicted SoH")
