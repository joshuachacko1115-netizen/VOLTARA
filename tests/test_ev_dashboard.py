"""EV assessment: logic in src/inference/ev_assess.py and the dashboard page that shows it."""
from pathlib import Path

import pandas as pd
import pytest

from src.config import MODELS_DIR, REPORTS_DIR
from src.inference import ev_assess as E

PAGE = str(Path(__file__).resolve().parents[1] / "dashboard" / "pages" / "ev_assessment.py")
needs_models = pytest.mark.skipif(
    not (MODELS_DIR / "ev_quick_soh.joblib").exists() or not (MODELS_DIR / "ev_degradation_soh.joblib").exists()
    or not (REPORTS_DIR / "results" / "ev_split_units.csv").exists(),
    reason="EV models not trained - run `python -m src.models.train_ev`")


def test_range_is_capacity_soh_soc_over_consumption():
    now, full = E.range_km(60, 90, 50, 18)
    assert full == pytest.approx(60 * 0.9 / 18 * 100) and now == pytest.approx(full / 2)
    assert E.range_km(60, 90, 50, 0) == (None, None)


def test_manual_input_validation():
    assert E.validate_manual(E.ManualInput()) == []
    errors = E.validate_manual(E.ManualInput(capacity_kwh=0, soc_pct=120))
    assert any("capacity" in e for e in errors) and any("Charge" in e for e in errors)


def test_dataset_detection_by_columns():
    assert E.detect_dataset(pd.DataFrame(columns=["battery_id", "observation_id", "cycle_count"])) == "ev_degradation"
    assert E.detect_dataset(pd.DataFrame(columns=["Vehicle_ID", "Trip_ID"])) == "ev_telematics"
    with pytest.raises(ValueError):
        E.detect_dataset(pd.DataFrame(columns=["cycle_number", "capacity_ah"]))


def test_insights_follow_the_numbers():
    a = E.Assessment("t", 82.0, 0.6, 200.0, 250.0, 100.0, 0.5, 250.0, anomalies=["x"], anomaly_assessed=True)
    text = " ".join(E.build_insights(a, trend=(95.0, 82.0, 4.0, 3.25), consumption=17, soc=80,
                                     usage={"fast_charge_ratio": 0.7, "avg_soc": 90}))
    assert "3.2 points per year" in text and "0.6 years" in text       # (82 - 80) / 3.25
    assert "within 5 points of end of life" in text and "DC fast" in text and "20 and 80" in text
    assert "1 finding" in text and "200 km" in text


@needs_models
def test_manual_assessment_uses_the_quick_models():
    inp = E.ManualInput(capacity_kwh=60, age_years=3, odometer_km=60000, consumption_kwh_100km=18, chemistry="NMC")
    a = E.assess_manual(inp)
    efc = 60000 * 18 / 100 / 60
    assert a.notes["equivalent_full_cycles"] == pytest.approx(efc)
    X = E.quick_features(3 * 365.25, efc, "NMC", "Unknown")
    soh, rul = E.predict_quick(E.load_quick_models(), X)
    assert a.soh_pct == pytest.approx(soh[0]) and a.rul_cycles == pytest.approx(rul[0])
    assert a.range_km == pytest.approx(60 * soh[0] / 100 * 0.8 / 18 * 100)
    assert not a.anomaly_assessed
    unknown = E.assess_manual(E.ManualInput(chemistry="Unknown"))
    assert unknown.rul_cycles is None and any("RUL is not shown" in w for w in unknown.warnings)
    scooter = E.assess_manual(E.ManualInput(vehicle_type="Electric scooter", capacity_kwh=3,
                                            consumption_kwh_100km=3.5, odometer_km=8000))
    assert any("extrapolation" in w for w in scooter.warnings)


@needs_models
@pytest.mark.parametrize("dataset", ["ev_degradation", "ev_telematics"])
def test_history_assessment_on_a_held_out_sample(dataset):
    unit = E.sample_units(dataset)[0]
    a = E.assess_history(E.sample_history(dataset, unit))
    assert 0 < a.soh_pct <= 100 and a.soh_measured_pct is not None
    assert a.anomaly_assessed and a.insights and a.range_km > 0
    assert len(a.history) == len(E.sample_history(dataset, unit))


@needs_models
def test_history_rejects_two_units():
    ds = "ev_degradation"
    u = E.sample_units(ds)[:2]
    with pytest.raises(ValueError, match="one battery_id"):
        E.assess_history(pd.concat([E.sample_history(ds, x) for x in u]))


@needs_models
def test_page_shows_the_assessment_module_output():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    at.radio[0].set_value("Enter vehicle information manually").run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    a = at.session_state["ev_result"]["a"]
    direct = E.assess_manual(E.ManualInput())
    assert a.soh_pct == pytest.approx(direct.soh_pct)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Battery health (SOH)"] == f"{direct.soh_pct:.1f}%"
    assert metrics["Remaining range"] == f"{direct.range_km:.0f} km"
    assert metrics["Battery anomalies"] == "Not assessed"

    at.radio[0].set_value("Use a sample battery dataset").run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    unit = E.sample_units("ev_degradation")[0]
    direct = E.assess_history(E.sample_history("ev_degradation", unit), 80.0, 17.0, "use")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Battery health (SOH)"] == f"{direct.soh_pct:.1f}%"
    assert metrics["Battery anomalies"] == ("Detected" if direct.anomalies else "None")


@needs_models
def test_assessment_reports_intervals_around_its_estimates():
    a = E.assess_manual(E.ManualInput(chemistry="NMC"))
    if a.soh_interval is None:
        pytest.skip("intervals not calibrated - run `python -m src.models.ev_intervals`")
    assert a.soh_interval[0] <= a.soh_pct <= a.soh_interval[1] <= 100
    assert a.rul_interval[0] <= a.rul_cycles <= a.rul_interval[1]
    unknown = E.assess_manual(E.ManualInput(chemistry="Unknown"))
    assert unknown.rul_interval is None and unknown.soh_interval is not None
    h = E.assess_history(E.sample_history("ev_degradation", E.sample_units("ev_degradation")[0]))
    assert h.soh_interval[0] <= h.soh_pct <= h.soh_interval[1]
