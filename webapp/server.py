"""VOLTARA web app: EV Intelligence vehicle assessment with a custom front end.

    .venv/bin/python -m uvicorn webapp.server:app --port 8600

Every number on the page comes from src.inference.ev_assess, the same code the Streamlit
EV page uses. This module only parses requests, turns an `Assessment` into JSON and
serves webapp/static/.

    GET  /                      the page
    GET  /api/meta              form defaults, sample batteries / vehicles, model errors
    GET  /api/sample            profile of one held-out sample (fills the form)
    POST /api/profile           profile of an uploaded history CSV
    POST /api/assess            run the assessment (manual, sample or upload)
    GET  /api/example.csv       a sample history in the dataset's original format
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from dashboard.helpers import read_history_csv
from src.inference import ev_assess as E
from src.inference.ev_predict import EVPredictor

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 5_000_000
DATASETS = {
    "ev_degradation": {"label": "Battery packs", "detail": "Health observations of fleet battery packs"},
    "ev_telematics": {"label": "Vehicle trips", "detail": "Trip telematics of individual vehicles"},
}


# --------------------------------------------------------------------------- #
# Cached model and data access
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def quick_models() -> dict:
    return E.load_quick_models()


@lru_cache(maxsize=2)
def predictor(dataset: str) -> EVPredictor:
    return EVPredictor(dataset)


@lru_cache(maxsize=2)
def samples(dataset: str) -> tuple[str, ...]:
    return tuple(E.sample_units(dataset))


@lru_cache(maxsize=256)
def _sample(dataset: str, unit: str) -> pd.DataFrame:
    return E.sample_history(dataset, unit)


def sample(dataset: str, unit: str) -> pd.DataFrame:
    if dataset not in DATASETS or unit not in samples(dataset):
        raise ValueError(f"Unknown sample {dataset}/{unit}.")
    return _sample(dataset, unit).copy()


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, pd.DataFrame):
        return jsonable(v.to_dict(orient="records"))
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    if v is None or isinstance(v, (str, int)):
        return v
    if v is pd.NaT or (not isinstance(v, str) and pd.isna(v)):
        return None
    return str(v)


def assessment_json(a: E.Assessment) -> dict:
    return jsonable({k: getattr(a, k) for k in a.__dataclass_fields__})


def error(msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


# --------------------------------------------------------------------------- #
# Profile: what the form shows for a history (sample or upload)
# --------------------------------------------------------------------------- #
def profile(raw: pd.DataFrame) -> dict:
    """Values of the latest row, for display in the form. The models read the full history,
    not these numbers; `soc_pct` / `consumption_kwh_100km` are what the range uses by default."""
    dataset = E.detect_dataset(raw)
    unit_col = E.raw_unit_column(dataset)
    if raw[unit_col].nunique() != 1:
        raise ValueError(f"The CSV must contain one {unit_col}; found {raw[unit_col].nunique()}.")
    f = predictor(dataset).prepare(raw)
    last = f.iloc[-1]
    if dataset == "ev_degradation":
        age_days = float(last.battery_age_days)
        out = {"capacity_kwh": last.pack_capacity_kwh, "age_years": age_days / 365.25,
               # The data have no odometer: average daily distance x age.
               "odometer_km": last.avg_daily_distance_km * age_days, "odometer_note": "avg daily distance × age",
               "soc_pct": last.avg_state_of_charge_pct, "soc_note": "Average charge recorded in the data.",
               "consumption_kwh_100km": None,
               "consumption_note": "Not recorded in this dataset: the car default is used.",
               "chemistry": last.chemistry, "vehicle": f"{last.manufacturer} {last.vehicle_model}",
               "cycles": last.efc, "unit": last.battery_id}
    else:
        consumption = float(last.wh_per_km) / 10
        # Age from the mid-model-year; a trip early in the model year can give a few weeks below zero.
        out = {"capacity_kwh": last.battery_capacity_kwh, "age_years": max(float(last.vehicle_age_years), 0.0),
               # No odometer either: cycle count x capacity / consumption.
               "odometer_km": last.battery_cycles * last.battery_capacity_kwh / consumption * 100,
               "odometer_note": "cycles × capacity ÷ consumption",
               "soc_pct": last.soc_pct, "soc_note": "Charge on the last trip in the data.",
               "consumption_kwh_100km": consumption, "consumption_note": "Consumption on the last trip in the data.",
               "chemistry": last.battery_type, "vehicle": f"{last.manufacturer} {last.model}",
               "cycles": last.battery_cycles, "unit": last.vehicle_id}
    out.update(dataset=dataset, dataset_label=DATASETS[dataset]["label"], rows=len(f),
               default_consumption_kwh_100km=E.VEHICLE_TYPES["Electric car"]["consumption_kwh_100km"])
    return jsonable(out)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
async def index(request: Request) -> Response:
    return FileResponse(STATIC / "index.html")


def meta(request: Request) -> Response:
    models = quick_models()
    soh, rul = models["quick_soh"]["meta"], models["quick_rul"]["meta"]
    return JSONResponse(jsonable({
        "vehicle_types": E.VEHICLE_TYPES, "chemistries": E.CHEMISTRIES, "climates": E.CLIMATES,
        "datasets": {k: {**v, "units": list(samples(k))} for k, v in DATASETS.items()},
        "eol_soh_pct": E.EV_EOL_SOH_PCT,
        "errors": {"quick_soh": soh["test_scores"]["MAE"], "quick_rul": rul["test_scores"]["MAE"],
                   "degradation_soh": E.model_error("degradation_soh"),
                   "degradation_rul": E.model_error("degradation_rul"),
                   "telematics_soh": E.model_error("telematics_soh")},
    }))


def sample_profile(request: Request) -> Response:
    try:
        raw = sample(request.query_params.get("dataset", ""), request.query_params.get("unit", ""))
        return JSONResponse(profile(raw))
    except ValueError as e:
        return error(str(e))


async def upload_profile(request: Request) -> Response:
    try:
        raw = await _upload(request)
        return JSONResponse(await run_in_threadpool(profile, raw))
    except ValueError as e:
        return error(str(e))


async def _upload(request: Request) -> pd.DataFrame:
    body = await request.body()
    if len(body) > MAX_UPLOAD_BYTES:
        raise ValueError(f"The file is larger than {MAX_UPLOAD_BYTES // 1_000_000} MB.")
    return read_history_csv(body)


def _num(body: dict, key: str) -> float | None:
    v = body.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number.") from None


def range_inputs(a: E.Assessment, capacity_kwh: float) -> dict | None:
    """The charge and consumption the range used (it inverts `ev_assess.range_km`)."""
    if not a.range_km or not a.full_range_km:
        return None
    return {"capacity_kwh": capacity_kwh, "soc_pct": 100 * a.range_km / a.full_range_km,
            "consumption_kwh_100km": capacity_kwh * a.soh_pct / a.full_range_km,
            "new_range_km": a.full_range_km / (a.soh_pct / 100)}


def run_assessment(body: dict) -> dict:
    source = body.get("source")
    if source == "manual":
        inp = E.ManualInput(
            vehicle_type=body.get("vehicle_type", "Electric car"), capacity_kwh=_num(body, "capacity_kwh"),
            soc_pct=_num(body, "soc_pct"), age_years=_num(body, "age_years"), odometer_km=_num(body, "odometer_km"),
            consumption_kwh_100km=_num(body, "consumption_kwh_100km"),
            chemistry=body.get("chemistry") or "Unknown", climate=body.get("climate") or "Unknown")
        if inp.vehicle_type not in E.VEHICLE_TYPES or inp.chemistry not in E.CHEMISTRIES \
                or inp.climate not in E.CLIMATES:
            raise ValueError("Unknown vehicle type, chemistry or climate.")
        a = E.assess_manual(inp, quick_models())
        return {"assessment": assessment_json(a), "profile": None,
                "range": jsonable(range_inputs(a, inp.capacity_kwh))}
    if source == "sample":
        raw = sample(body.get("dataset", ""), body.get("unit", ""))
    elif source == "upload":
        csv = body.get("csv") or ""
        if len(csv.encode()) > MAX_UPLOAD_BYTES:
            raise ValueError(f"The file is larger than {MAX_UPLOAD_BYTES // 1_000_000} MB.")
        raw = read_history_csv(csv.encode())
    else:
        raise ValueError("source must be manual, sample or upload.")
    # None = use what the data say (see `profile`); a number = the value typed in the form.
    a = E.assess_history(raw, _num(body, "soc_pct"), _num(body, "consumption_kwh_100km"), source)
    prof = profile(raw)
    return {"assessment": assessment_json(a), "profile": prof, "range": jsonable(range_inputs(a, prof["capacity_kwh"]))}


async def assess(request: Request) -> Response:
    try:
        body = await request.json()
    except ValueError:
        return error("The request body must be JSON.")
    if not isinstance(body, dict):
        return error("The request body must be a JSON object.")
    try:
        return JSONResponse(await run_in_threadpool(run_assessment, body))
    except (ValueError, KeyError) as e:
        return error(str(e).strip("'\""))


def example_csv(request: Request) -> Response:
    dataset = request.query_params.get("dataset", "ev_degradation")
    if dataset not in DATASETS:
        return error("Unknown dataset.")
    raw = sample(dataset, samples(dataset)[0])
    return Response(raw.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="voltara_{dataset}_example.csv"'})


app = Starlette(routes=[
    Route("/", index),
    Route("/api/meta", meta),
    Route("/api/sample", sample_profile),
    Route("/api/profile", upload_profile, methods=["POST"]),
    Route("/api/assess", assess, methods=["POST"]),
    Route("/api/example.csv", example_csv),
    Mount("/static", StaticFiles(directory=STATIC), name="static"),
])
