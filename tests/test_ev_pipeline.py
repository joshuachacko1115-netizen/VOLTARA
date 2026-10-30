import json

import numpy as np
import pandas as pd
import pytest

from src.config import EV_DIR, MODELS_DIR
from src.features.ev_features import ev1_features, ev2_features
from src.models.train_ev import build_tasks, choose_test_units
from src.preprocessing.ev_cleaning import (OutlierMasker, clean_ev_degradation, clean_ev_telematics,
                                           fill_static)


def _ev1_frame(n_batt=3, n_obs=8, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    k = 0
    for b in range(n_batt):
        made = pd.Timestamp("2022-01-01") + pd.Timedelta(days=30 * b)
        for j in range(n_obs):
            k += 1
            age = 60 * (j + 1)
            soh = 100 - 0.01 * age + rng.normal(0, 0.3)
            rows.append({
                "battery_id": f"BATT_{b:05d}", "observation_id": f"OBS_{k:06d}", "chemistry": "NMC",
                "manufacturer": "CellTech", "vehicle_model": "Van-E4", "pack_capacity_kwh": 70.0,
                "manufacture_date": made, "observation_date": made + pd.Timedelta(days=age),
                "battery_age_days": float(age), "cycle_count": age * 0.8, "climate_zone": "mixed",
                "avg_daily_distance_km": 40.0, "avg_charging_power_kw": 30.0 + rng.normal(0, 2),
                "fast_charge_ratio": 0.3, "avg_depth_of_discharge_pct": 60.0, "avg_state_of_charge_pct": 65.0,
                "charge_cycles_per_month": 20.0, "total_energy_throughput_kwh": age * 0.8 * 65,
                "avg_ambient_temp_c": 20 + rng.normal(0, 3), "humidity_pct": 50.0, "altitude_m": 100.0,
                "internal_resistance_mohm": 40 + 0.02 * age + rng.normal(0, 1), "voltage_avg_v": 360.0,
                "current_avg_a": 140 + rng.normal(0, 5), "charge_time_avg_min": 90 + rng.normal(0, 10),
                "peak_temp_during_fast_charge_c": 30 + rng.normal(0, 3), "capacity_fade_pct": 100 - soh,
                "service_count": 0.0, "last_service_date": pd.NaT, "bms_firmware_version": "BMS_v3.0",
                "recall_flag": 0.0, "state_of_health_pct": soh, "remaining_useful_life_cycles": 1000 - j,
                "failure_risk_label": "Low", "degradation_category": "Normal"})
    return pd.DataFrame(rows)


def test_leak_column_is_renamed_and_never_a_model_input():
    clean, rep = clean_ev_degradation(_ev1_frame())
    assert "capacity_fade_pct" not in clean and "leak_capacity_fade_pct" in clean
    assert rep["capacity_fade_equals_100_minus_soh"]
    for task in build_tasks(ev1_features(clean)[1], ["soc_pct"]):
        inputs = set(task.nums + task.cats)
        assert not any(f.startswith("leak_") for f in inputs)
        assert not inputs & {"state_of_health_pct", "remaining_useful_life_cycles", "failure_risk_label",
                             "degradation_category", "soh_pct", "failure_type", "maintenance_required",
                             "fire_risk", "fire_risk_2", "wh_per_km", "range_remaining_km"}
    consumption = [t for t in build_tasks(["x"], ["x"]) if t.key == "telematics_consumption"][0]
    # Energy, current and battery temperature are computed from the trip's energy use.
    assert not set(consumption.nums) & {"energy_consumed_kwh", "battery_current_a", "battery_temp_c"}


def test_static_attributes_and_age_are_recovered():
    df = _ev1_frame()
    df.loc[[1, 9], "chemistry"] = np.nan
    df.loc[2, "battery_age_days"] = np.nan
    df.loc[3, "observation_date"] = pd.NaT
    clean, rep = clean_ev_degradation(df)
    assert clean.chemistry.notna().all() and rep["filled_static_chemistry"] == 2
    assert clean.loc[2, "battery_age_days"] == 180
    assert clean.loc[3, "observation_date"] == df.loc[3, "manufacture_date"] + pd.Timedelta(days=240)


def test_static_conflict_raises():
    df = pd.DataFrame({"u": ["a", "a"], "chem": ["NMC", "LFP"]})
    with pytest.raises(ValueError):
        fill_static(df, "u", ["chem"], {})


def test_implausible_values_become_missing():
    df = _ev1_frame()
    df.loc[4, "avg_ambient_temp_c"] = 83.7
    df.loc[5, "fast_charge_ratio"] = 1.4
    clean, _ = clean_ev_degradation(df)
    assert np.isnan(clean.loc[4, "avg_ambient_temp_c"]) and np.isnan(clean.loc[5, "fast_charge_ratio"])


def test_outlier_flags_spike_not_trend_and_are_causal():
    # Rising resistance; noise about the size of the per-step rise, as in ev_degradation
    # (fitted log-residual spread 0.11 vs typical drift 0.09).
    t = np.arange(30.0)
    df = pd.DataFrame({"battery_id": "b", "r": 40 + 0.5 * t + np.random.default_rng(1).normal(0, 0.7, 30)})
    df.loc[20, "r"] *= 2.5
    m = OutlierMasker(["r"], {"r"}, window=5, n_sigma=4.0).fit(df)
    fl = m.flags(df)["r"].to_numpy()
    assert fl[20] and not fl[:20].any()          # the spike, not the rising trend before it
    # Known limitation (OutlierMasker docstring): for the next `window` rows the spike sits in
    # the reference window and can flag a normal reading. Excluding flagged readings from the
    # reference was tested and lowered precision on ev_degradation (0.77 -> 0.73).
    assert fl[21:26].sum() <= 2 and not fl[26:].any()
    assert (m.flags(df.iloc[:15])["r"].to_numpy() == fl[:15]).all()   # later rows change nothing earlier


def test_ev1_features_are_causal():
    clean, _ = clean_ev_degradation(_ev1_frame())
    full, cols = ev1_features(clean)
    cut = clean.groupby("battery_id").head(5)          # first 5 observations of every battery
    part, _ = ev1_features(cut.reset_index(drop=True))
    a = full.set_index("observation_id").loc[part.observation_id, cols]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), part[cols].reset_index(drop=True))


def test_ev2_features_are_causal():
    rng = np.random.default_rng(0)
    n = 12
    df = pd.DataFrame({
        "vehicle_id": ["V1"] * 6 + ["V2"] * 6, "trip_id": [f"T{i}" for i in range(n)], "driver_id": "D1",
        "manufacturer": "Tata", "model": "Nexon EV", "year": 2022, "battery_capacity_kwh": 45.0,
        "battery_type": "LFP", "motor_power_kw": 106, "vehicle_weight_kg": 1500,
        "trip_date": pd.date_range("2024-01-01", periods=n, freq="20D"),
        "distance_km": rng.uniform(10, 80, n), "avg_speed": rng.uniform(20, 90, n),
        "ambient_temp_c": rng.uniform(10, 35, n), "weather": "Sunny", "road_type": "City",
        "battery_voltage_v": 355.0, "battery_current_a": rng.uniform(10, 60, n),
        "battery_temp_c": rng.uniform(15, 40, n), "soc_pct": 50, "soh_pct": 97,
        "battery_cycles": np.arange(n) * 10 + 5, "charging_power_kw": rng.choice([7.2, 50.0], n),
        "charging_duration_min": 60, "energy_consumed_kwh": rng.uniform(2, 15, n),
        "range_remaining_km": 100.0, "driving_style": rng.choice(["Calm", "Aggressive"], n),
        "failure_type": "Normal", "maintenance_required": "No", "fire_risk": "Low"})
    clean, rep = clean_ev_telematics(df)
    assert "leak_range_remaining_km" in clean and "range_remaining_km" not in clean
    full, cols = ev2_features(clean)
    part, _ = ev2_features(clean.groupby("vehicle_id").head(3).reset_index(drop=True))
    a = full.set_index("trip_id").loc[part.trip_id, cols]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), part[cols].reset_index(drop=True))


def test_test_units_are_disjoint_deterministic_and_stratified():
    df = pd.DataFrame({"u": [f"u{i}" for i in range(100)], "chem": ["A"] * 70 + ["B"] * 30})
    t1, t2 = choose_test_units(df, "u", "chem"), choose_test_units(df, "u", "chem")
    assert t1 == t2 and len(t1) == 20
    assert sum(u in t1 for u in df.u[df.chem == "B"]) == 6


# --------------------------------------------------------------------------- #
# Saved models (need `python -m src.models.train_ev` first)
# --------------------------------------------------------------------------- #
needs_models = pytest.mark.skipif(not (MODELS_DIR / "ev_degradation_soh.joblib").exists()
                                  or not (EV_DIR / "ev_degradation_features.parquet").exists(),
                                  reason="EV models not trained")


@needs_models
@pytest.mark.parametrize("task,table", [("degradation_soh", "ev_degradation"), ("telematics_soh", "ev_telematics")])
def test_saved_model_reproduces_reported_test_score(task, table):
    import joblib
    obj = joblib.load(MODELS_DIR / f"ev_{task}.joblib")
    meta = obj["meta"]
    df = pd.read_parquet(EV_DIR / f"{table}_features.parquet")
    test = df[df.split == "test"]
    dev_units = set(df.loc[df.split == "dev", meta["unit"]])
    assert not dev_units & set(test[meta["unit"]])                       # no unit in both
    pred = obj["pipeline"].predict(test[meta["categorical_features"] + meta["numeric_features"]])
    mae = np.mean(np.abs(pred - test[meta["target"]].to_numpy()))
    assert abs(mae - meta["test_scores"]["MAE"]) < 1e-9
    assert json.loads((MODELS_DIR / f"ev_{task}.json").read_text())["test_scores"]["MAE"] == pytest.approx(mae)


@needs_models
def test_ev_predictor_matches_training_features_on_raw_rows():
    from src.data.ev_loaders import find_ev_files
    from src.inference.ev_predict import EVPredictor
    import joblib
    raw = pd.read_csv(find_ev_files()["ev_degradation"])
    unit = sorted(raw.battery_id.unique())[0]
    out = EVPredictor("ev_degradation").predict(raw[raw.battery_id == unit])
    obj = joblib.load(MODELS_DIR / "ev_degradation_soh.joblib")
    feats = pd.read_parquet(EV_DIR / "ev_degradation_features.parquet").query("battery_id == @unit")
    X = feats[obj["meta"]["categorical_features"] + obj["meta"]["numeric_features"]]
    np.testing.assert_allclose(out.pred_degradation_soh.to_numpy(), obj["pipeline"].predict(X), rtol=0, atol=1e-9)
    assert {"pred_degradation_rul", "pred_degradation_risk", "p_degradation_risk_High"} <= set(out.columns)


def test_conformal_interval_covers_and_widens_by_bin():
    from src.models.common import apply_interval
    from src.models.ev_intervals import check, fit_interval
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 100, 4000)
    y = pred + rng.normal(0, 1 + pred / 25)          # noise grows with the prediction
    iv = fit_interval(pred[:2000], y[:2000])
    assert iv["half_width"] == sorted(iv["half_width"])  # wider where errors are larger
    res = check(iv, pred[2000:], y[2000:], (None, None))
    assert 0.87 < res["test_coverage"] < 0.93
    lo, hi = apply_interval(iv, [50.0, 99.0], (0.0, 100.0))
    assert (lo < [50.0, 99.0]).all() and hi[1] == 100.0


@needs_models
@pytest.mark.parametrize("task", ["degradation_soh", "degradation_rul", "telematics_soh", "quick_soh", "quick_rul"])
def test_saved_intervals_reach_their_coverage_on_test_units(task):
    meta = json.loads((MODELS_DIR / f"ev_{task}.json").read_text())
    if "interval" not in meta:
        pytest.skip("intervals not calibrated - run `python -m src.models.ev_intervals`")
    for key in ("interval", "interval_categories_unknown"):
        if key in meta:
            assert meta[key]["level"] == 0.9 and abs(meta[key]["test_coverage"] - 0.9) < 0.04
