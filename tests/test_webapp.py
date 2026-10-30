"""Web app (webapp/server.py): the API returns exactly what src.inference.ev_assess computes."""
import pytest

from src.config import MODELS_DIR, REPORTS_DIR
from src.inference import ev_assess as E

pytest.importorskip("starlette")
pytest.importorskip("httpx")
needs_models = pytest.mark.skipif(
    not (MODELS_DIR / "ev_quick_soh.joblib").exists() or not (MODELS_DIR / "ev_degradation_soh.joblib").exists()
    or not (REPORTS_DIR / "results" / "ev_split_units.csv").exists(),
    reason="EV models not trained - run `python -m src.models.train_ev`")


@pytest.fixture(scope="module")
def client():
    from starlette.testclient import TestClient
    from webapp.server import app
    return TestClient(app)


def test_page_and_assets_are_served(client):
    html = client.get("/").text
    for text in ("EV Intelligence — Vehicle Assessment", "Choose data source", "Current charge (SOC)",
                 "Preview expected output", "EV Intelligence Dashboard", "Battery anomalies"):
        assert text in html
    assert client.get("/static/app.js").status_code == 200 and client.get("/static/app.css").status_code == 200


@needs_models
def test_manual_matches_the_assessment_module(client):
    body = {"source": "manual", "vehicle_type": "Electric car", "capacity_kwh": 60, "soc_pct": 80, "age_years": 3,
            "odometer_km": 45000, "consumption_kwh_100km": 16, "chemistry": "NMC", "climate": "Unknown"}
    res = client.post("/api/assess", json=body).json()
    a = E.assess_manual(E.ManualInput("Electric car", 60, 80, 3, 45000, 16, "NMC", "Unknown"))
    got = res["assessment"]
    assert got["soh_pct"] == pytest.approx(a.soh_pct) and got["rul_cycles"] == pytest.approx(a.rul_cycles)
    assert got["range_km"] == pytest.approx(a.range_km) and got["insights"] == a.insights
    assert res["range"]["soc_pct"] == pytest.approx(80) and res["range"]["consumption_kwh_100km"] == pytest.approx(16)


@needs_models
@pytest.mark.parametrize("dataset", ["ev_degradation", "ev_telematics"])
def test_sample_and_upload_match_the_assessment_module(client, dataset):
    unit = E.sample_units(dataset)[0]
    raw = E.sample_history(dataset, unit)
    direct = E.assess_history(raw, None, None, "sample")
    res = client.post("/api/assess", json={"source": "sample", "dataset": dataset, "unit": unit}).json()
    a = res["assessment"]
    assert a["soh_pct"] == pytest.approx(direct.soh_pct) and a["rul_cycles"] == pytest.approx(direct.rul_cycles)
    assert a["range_km"] == pytest.approx(direct.range_km) and a["anomalies"] == direct.anomalies
    assert len(a["history"]) == len(raw) and res["profile"]["unit"] == unit
    up = client.post("/api/assess", json={"source": "upload", "csv": raw.to_csv(index=False)}).json()["assessment"]
    assert up["soh_pct"] == pytest.approx(direct.soh_pct) and up["anomalies"] == direct.anomalies
    # A typed charge and consumption change the range only.
    typed = client.post("/api/assess", json={"source": "sample", "dataset": dataset, "unit": unit,
                                             "soc_pct": 50, "consumption_kwh_100km": 20}).json()
    assert typed["assessment"]["soh_pct"] == pytest.approx(direct.soh_pct)
    assert typed["range"]["soc_pct"] == pytest.approx(50) and typed["range"]["consumption_kwh_100km"] == pytest.approx(20)


@needs_models
def test_bad_requests_get_readable_errors(client):
    r = client.post("/api/assess", json={"source": "manual", "capacity_kwh": 0, "soc_pct": 80, "age_years": 3,
                                         "odometer_km": 1000, "consumption_kwh_100km": 16})
    assert r.status_code == 400 and "capacity" in r.json()["error"]
    assert client.post("/api/assess", json={"source": "sample", "dataset": "ev_degradation", "unit": "nope"}).status_code == 400
    r = client.post("/api/profile", content=b"cycle_number,capacity_ah\n1,2\n")
    assert r.status_code == 400 and "Unrecognised CSV" in r.json()["error"]
