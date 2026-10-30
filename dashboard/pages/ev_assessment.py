"""VOLTARA — EV Intelligence: vehicle assessment page (second page of the dashboard).

Opened from the sidebar of `streamlit run dashboard/app.py`. Input on the left (sample
battery, CSV upload, or manual entry), outputs on the right, insights and the SoH trend
below. All numbers come from src.inference.ev_assess; this file only lays them out.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import html  # noqa: E402

import streamlit as st  # noqa: E402

from dashboard import helpers as H  # noqa: E402
from dashboard import style as S  # noqa: E402
from src.inference import ev_assess as E  # noqa: E402

st.set_page_config(page_title="VOLTARA — EV Assessment", page_icon="🚗", layout="wide")

SOURCES = ["Use a sample battery dataset", "Upload historical battery CSV", "Enter vehicle information manually"]
SAMPLE_SETS = {"ev_degradation": "EV fleet battery packs (health observations)",
               "ev_telematics": "EV trips (vehicle telematics)"}


@st.cache_resource(show_spinner="Loading models…")
def quick_models() -> dict:
    return E.load_quick_models()


@st.cache_data(show_spinner=False)
def samples(dataset: str) -> list[str]:
    return E.sample_units(dataset)


@st.cache_data(show_spinner=False)
def sample(dataset: str, unit: str):
    return E.sample_history(dataset, unit)


def assess(source: str, payload) -> tuple[E.Assessment | None, str | None]:
    try:
        if source == SOURCES[2]:
            return E.assess_manual(payload, quick_models()), None
        raw, soc, consumption, label = payload
        return E.assess_history(raw, soc, consumption, label), None
    except (ValueError, KeyError) as e:
        return None, str(e).strip("'\"")


st.markdown(S.CSS, unsafe_allow_html=True)
st.markdown(S.header("EV Intelligence · vehicle assessment: health, range, remaining life, anomalies"),
            unsafe_allow_html=True)
try:
    quick_models()
except Exception as e:  # missing, or saved by an older version of the training code
    st.error(f"The EV models could not be loaded ({type(e).__name__}: {e}). "
             "Run `python -m src.models.train_ev` and reload this page.")
    st.stop()

left_col, right_col = st.columns(2, gap="large")
left, right = left_col.container(border=True), right_col.container(border=True)

# --------------------------------------------------------------------------- #
# 1. Input
# --------------------------------------------------------------------------- #
with left:
    st.markdown(S.step(1, "Vehicle assessment"), unsafe_allow_html=True)
    source = st.radio("Choose data source", SOURCES)
    payload, key = None, source
    vehicle_type = st.radio("Vehicle type", list(E.VEHICLE_TYPES), horizontal=True)
    defaults = E.VEHICLE_TYPES[vehicle_type]

    if source == SOURCES[0]:
        st.caption("A battery from the held-out test set (never used for training) demonstrates the predictions; "
                   "its recorded SoH is shown next to the prediction.")
        dataset = st.selectbox("Sample dataset", list(SAMPLE_SETS), format_func=SAMPLE_SETS.get)
        unit = st.selectbox("Battery / vehicle", samples(dataset))
        raw = sample(dataset, unit)
        key = f"{source}-{dataset}-{unit}"
    elif source == SOURCES[1]:
        st.caption("One battery or vehicle, oldest row first, in either dataset's original column format.")
        up = st.file_uploader("Historical battery CSV", type=["csv"])
        c1, c2 = st.columns(2)
        for col, ds in zip((c1, c2), SAMPLE_SETS):
            first = samples(ds)[0]
            col.download_button(f"Example CSV: {ds.replace('ev_', '')}", sample(ds, first).to_csv(index=False),
                                file_name=f"voltara_{ds}_example.csv", mime="text/csv", key=f"dl-{ds}")
        raw = None
        if up is not None:
            try:
                raw = H.read_history_csv(up.getvalue())
                key = f"{source}-{up.name}-{up.size}"
            except ValueError as e:
                st.error(str(e))
    else:
        raw = None

    if source == SOURCES[2]:
        capacity = st.number_input("Battery capacity (kWh)", 0.5, 1000.0, float(defaults["capacity_kwh"]), 0.5,
                                   key=f"cap-{vehicle_type}")
        soc = st.slider("Current charge (SOC, %)", 0, 100, 80)
        a1, a2 = st.columns(2)
        age = a1.number_input("Vehicle age (years)", 0.0, 40.0, 3.0, 0.5)
        odo = a2.number_input("Odometer reading (km)", 0.0, 3_000_000.0, 45000.0, 1000.0)
        consumption = st.number_input("Average energy consumption (kWh/100 km)", 1.0, 400.0,
                                      float(defaults["consumption_kwh_100km"]), 0.5, key=f"cons-{vehicle_type}")
        c1, c2 = st.columns(2)
        chem = c1.selectbox("Cell chemistry", E.CHEMISTRIES,
                            help="Strongly affects the estimate; RUL is only shown when it is known.")
        climate = c2.selectbox("Operating climate (optional)", E.CLIMATES)
        payload = E.ManualInput(vehicle_type, capacity, float(soc), age, odo, consumption, chem, climate)
        key = repr(payload)
    elif raw is not None:
        with st.expander("Range settings", expanded=False):
            st.caption("Used for the range estimate only. Telematics data already contain the last trip's "
                       "charge and consumption; tick to override them.")
            override = st.checkbox("Use these values", value=False, key=f"ovr-{key}")
            soc = st.slider("Current charge (SOC, %)", 0, 100, 80, key=f"soc-{key}")
            consumption = st.number_input("Average energy consumption (kWh/100 km)", 1.0, 400.0,
                                          float(defaults["consumption_kwh_100km"]), 0.5, key=f"cons-{key}")
        use = override or E.detect_dataset_safe(raw) == "ev_degradation"
        payload = (raw, float(soc) if use else None, consumption if use else None, source.split()[0].lower())
        key = f"{key}-{use}-{soc}-{consumption}"

    clicked = st.button("Assess vehicle", type="primary", width="stretch", disabled=payload is None)

if clicked and payload is not None:
    a, err = assess(source, payload)
    st.session_state["ev_result"] = {"a": a, "error": err, "key": key, "vehicle_type": vehicle_type}
result = st.session_state.get("ev_result")

# --------------------------------------------------------------------------- #
# 2. Output
# --------------------------------------------------------------------------- #
with right:
    st.markdown(S.step(2, "Results"), unsafe_allow_html=True)
    if result is None:
        st.markdown(S.empty_state().replace("Pick a battery", "Choose a data source"), unsafe_allow_html=True)
    elif result["error"]:
        st.error(result["error"])
    else:
        a: E.Assessment = result["a"]
        if result["key"] != key:
            st.warning("The inputs changed after this assessment. Press **Assess vehicle** to update.")
        m1, m2 = st.columns(2)
        m1.metric("Battery health (SOH)", f"{a.soh_pct:.1f}%",
                  help=f"Model estimate · typical error ±{a.soh_error_pct:.2f} points on held-out batteries")
        m2.metric("Remaining range", "—" if a.range_km is None else f"{a.range_km:.0f} km",
                  help="Capacity × SOH × charge ÷ consumption: arithmetic under the stated consumption, "
                       "not a validated range model")
        m3, m4 = st.columns(2)
        if a.rul_cycles is None:
            rul_text = "—"
        elif a.soh_pct <= 80:
            rul_text = "End of life"
        else:
            rul_text = f"{a.rul_cycles:,.0f} cycles"
        m3.metric("Remaining useful life", rul_text,
                  help=f"Equivalent full cycles until 80 % SOH · typical error ±{a.rul_error_cycles:.0f} cycles")
        m4.metric("Battery anomalies", "Not assessed" if not a.anomaly_assessed
                  else ("Detected" if a.anomalies else "None"),
                  help="Rule checks and statistical outlier flags on the available readings")
        st.markdown(S.soh_bar(a.soh_pct / 100, "Estimated SOH"), unsafe_allow_html=True)
        notes = [f"Estimated model output · typical error ±{a.soh_error_pct:.1f} SOH points"]
        if a.soh_interval:
            notes.append(f"90 % range **{a.soh_interval[0]:.1f}–{a.soh_interval[1]:.1f}%**")
        if a.rul_interval:
            notes.append(f"RUL 90 % range **{a.rul_interval[0]:,.0f}–{a.rul_interval[1]:,.0f} cycles**")
        if a.soh_measured_pct is not None:
            notes.append(f"recorded SOH in the dataset: **{a.soh_measured_pct:.1f}%**")
        if a.rul_years is not None and a.soh_pct > 80:
            notes.append(f"RUL ≈ **{a.rul_years:.1f} years** at this vehicle's cycling rate so far")
        st.caption(" · ".join(notes))
        if a.notes.get("rul_note"):
            st.caption(a.notes["rul_note"])
        if a.risk:
            st.markdown(H.risk_badge(a.risk, "Degradation risk: "), unsafe_allow_html=True)
        if "p_failure_any" in a.notes:
            st.caption(f"Failure-type model: {a.notes['p_failure_any']:.0%} chance of a non-normal failure label; "
                       f"maintenance model: {a.notes['p_maintenance']:.0%}. These labels are largely random in "
                       "the training data (AUC 0.6–0.9), so treat them as weak signals.")
        if not a.anomaly_assessed:
            st.caption("Anomalies need a history of readings: upload a CSV or use a sample.")
        elif a.anomalies:
            st.error(f"{len(a.anomalies)} finding(s). Latest: {a.anomalies[0]}")
        for w in a.warnings:
            st.warning(w)

# --------------------------------------------------------------------------- #
# 3. Insights and trend
# --------------------------------------------------------------------------- #
if result is not None and result["a"] is not None:
    a = result["a"]
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    box = st.container(border=True)
    box.markdown(S.step(3, "Insights"), unsafe_allow_html=True)
    i1, i2 = box.columns([1.1, 1], gap="large")
    i1.markdown('<ul class="v-insights">' + "".join(f"<li>{html.escape(t)}</li>" for t in a.insights) + "</ul>",
                unsafe_allow_html=True)
    i1.caption("Generated by fixed rules from the numbers above (not a language model). "
               "SOH and RUL are model estimates; range uses the stated consumption; thresholds are project choices.")
    if a.history is not None and len(a.history) > 1:
        i2.markdown("**Battery degradation trend**")
        i2.altair_chart(H.ev_soh_chart(a.history), width="stretch")
    if a.anomalies:
        with box.expander(f"All anomaly findings ({len(a.anomalies)})"):
            for m in a.anomalies:
                st.markdown(f"- {m}")
    if a.history is not None:
        with box.expander("Per-row predictions"):
            st.dataframe(a.history, width="stretch", hide_index=True)
