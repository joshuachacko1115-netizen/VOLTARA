"""VOLTARA — Battery Intelligence dashboard (local demo).

Run from the project folder, with the venv active:
    streamlit run dashboard/app.py
(`python dashboard/app.py` also works: it relaunches itself under Streamlit.)

Input on the left, output on the right, charts below. All predictions come from
src.inference.predict.VoltaraPredictor.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `streamlit run` puts dashboard/, not the project root, on sys.path
    sys.path.insert(0, str(ROOT))

try:
    import streamlit as st
except ModuleNotFoundError:  # wrong interpreter (e.g. VS Code's Run button outside the venv)
    sys.exit(f"Streamlit is not installed for {sys.executable}.\n"
             "Activate the project venv first:  source .venv/bin/activate   (Windows: .venv\\Scripts\\activate)\n"
             "then run:  streamlit run dashboard/app.py")

if __name__ == "__main__" and not st.runtime.exists():
    # Started as `python dashboard/app.py` (or the editor's Run button): relaunch under Streamlit.
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    sys.exit(stcli.main())

import pandas as pd  # noqa: E402

from dashboard import helpers as H  # noqa: E402
from dashboard import style as S  # noqa: E402
from src.inference.predict import (EOL_NOTE, RUL_VARIANTS, VoltaraPredictor,  # noqa: E402
                                   example_cells, example_history)

st.set_page_config(page_title="VOLTARA — Battery Intelligence", page_icon="🔋", layout="wide")


@st.cache_resource(show_spinner="Loading models…")
def get_predictor(dataset: str, rul_variant: str) -> VoltaraPredictor:
    return VoltaraPredictor(dataset, rul_variant=rul_variant)


@st.cache_data(show_spinner=False)
def get_example(dataset: str, cell: str) -> pd.DataFrame:
    return example_history(dataset, cell)


@st.cache_data(show_spinner=False)
def full_prediction(dataset: str, history: pd.DataFrame) -> pd.DataFrame | None:
    """Prediction on the whole history, used to place the "Predict as of" slider."""
    pred, _ = H.predict_or_error(get_predictor(dataset, "default"), history, None)
    return pred


@st.cache_data(show_spinner="Loading example batteries…")
def get_examples(dataset: str) -> list[str]:
    try:
        cells = example_cells(dataset)
    except (FileNotFoundError, OSError):
        return []
    keep = []
    for cell in cells:
        pred = full_prediction(dataset, get_example(dataset, cell))
        eol = H.end_of_life_index(pred)
        if eol is None or pred[get_predictor(dataset, "default").order_col].iloc[eol] >= H.MIN_LIFE:
            keep.append(cell)
    return keep


st.markdown(S.CSS, unsafe_allow_html=True)
st.markdown(S.header(), unsafe_allow_html=True)

try:
    get_predictor("battery_aging", "default")
except FileNotFoundError as e:
    st.error(f"Trained models are missing ({e.filename}). Run `python -m src.run_pipeline` first.")
    st.stop()

left_col, right_col = st.columns(2, gap="large")
left = left_col.container(border=True)
right = right_col.container(border=True)

# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
with left:
    st.markdown(S.step(1, "Input"), unsafe_allow_html=True)
    dataset = st.selectbox("Battery dataset", list(H.DATASET_LABELS), format_func=H.DATASET_LABELS.get)
    predictor = get_predictor(dataset, "default")
    ageing_col = predictor.order_col
    ageing_label = H.AGEING_LABELS[ageing_col]
    columns = VoltaraPredictor.input_columns(dataset)
    examples = get_examples(dataset)
    sources = (["Example NASA battery"] if examples else []) + ["Upload CSV", "Blank template"]
    source = st.radio("Measurements from", sources, horizontal=True)

    full, full_key = None, source
    if source == "Example NASA battery":
        cell = st.selectbox("Example battery", examples)
        full, full_key = get_example(dataset, cell), f"{source}-{cell}"
    elif source == "Upload CSV":
        up = st.file_uploader("History CSV (one row per cycle or check, oldest first)", type=["csv"])
        st.download_button("Download a blank template",
                           H.blank_template(columns, ageing_col).to_csv(index=False),
                           file_name=f"voltara_{dataset}_template.csv", mime="text/csv")
        if up is not None:
            try:
                full, full_key = H.read_history_csv(up.getvalue()), f"{source}-{up.name}-{up.size}"
            except ValueError as e:
                st.error(str(e))
    else:
        n = st.number_input("Rows", min_value=1, max_value=500, value=12, step=1)
        full, full_key = H.blank_template(columns, ageing_col, int(n)), f"{source}-{n}"

    # Predict as of a chosen row. Most NASA histories continue past end of life, where RUL is 0.
    base, base_key, eol_at = full, full_key, None
    whole = None if full is None or source == "Blank template" or len(full) < 2 else full_prediction(dataset, full)
    if whole is not None:
        eol = H.end_of_life_index(whole)
        eol_at = None if eol is None else float(whole[ageing_col].iloc[eol])
        ages = whole[ageing_col].round(1).tolist()
        upto = st.select_slider(
            f"Predict as of: {ageing_label.lower()}", options=list(range(len(ages))),
            value=H.default_cutoff(whole), format_func=lambda i: f"{ages[i]:g}", key=f"asof-{dataset}-{full_key}",
            help="The prediction uses the history up to this point, as if it were the latest measurement.")
        if eol_at is not None:
            st.caption(f"This battery reached end of life (80 % SoH) at {ageing_label.lower()} {eol_at:g}.")
        base = full.iloc[:upto + 1].reset_index(drop=True)
        base_key = f"{full_key}-{upto}"

    history = None
    if base is not None:
        with st.expander(f"Measurements · {len(base)} rows", expanded=source == "Blank template"):
            help_ = H.column_help(list(base.columns))
            history = st.data_editor(
                base, num_rows="dynamic", width="stretch", height=min(320, 36 * (len(base) + 2)),
                key=f"editor-{dataset}-{base_key}",
                column_config={c: st.column_config.Column(c, help=h) for c, h in help_.items()})
            extra = H.ignored_columns(history, columns)
            if extra:
                st.caption(f"Ignored columns: {', '.join(extra[:8])}" + (" …" if len(extra) > 8 else ""))

    with st.expander("Model options"):
        rul_variant = st.radio(
            "RUL model", list(RUL_VARIANTS), horizontal=True,
            format_func={"default": "Default", "censoring_aware": "Censoring-aware"}.get,
            help="Default: trained on cells that reached end of life. Censoring-aware: also uses cells that "
                 "never failed; fewer underestimates for slow-fading batteries, larger error on the others.")

    clicked = st.button("Predict", type="primary", width="stretch", disabled=history is None)

current = None if history is None else H.fingerprint(dataset, history, rul_variant, None)
if clicked and history is not None:
    model = get_predictor(dataset, rul_variant)
    pred, error = H.predict_or_error(model, history, None)
    # actual RUL is known when the full history goes on to reach end of life after the cut-off
    actual = None
    if pred is not None and eol_at is not None and len(history) == len(base) \
            and float(pred[ageing_col].iloc[-1]) < eol_at:
        actual = eol_at - float(pred[ageing_col].iloc[-1])
    st.session_state["result"] = {"pred": pred, "error": error, "dataset": dataset, "rul_variant": rul_variant,
                                  "fingerprint": current, "eol_at": eol_at, "actual_rul": actual}

result = st.session_state.get("result")

# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
with right:
    st.markdown(S.step(2, "Output"), unsafe_allow_html=True)
    if result is None:
        st.markdown(S.empty_state(), unsafe_allow_html=True)
    elif result["error"]:
        st.error(result["error"])
    else:
        if result["fingerprint"] != current:
            st.warning("The inputs changed after this prediction. Press **Predict** to update.")
        model = get_predictor(result["dataset"], result["rul_variant"])
        s = H.latest_summary(result["pred"], model)
        unit = s["rul_unit"]
        st.caption(f"As of {H.AGEING_LABELS[s['ageing_col']].lower()} {s['ageing']:g} · {s['rows']} rows")
        cover = f"{s['soh_coverage']:.0%} of the SoH inputs and {s['rul_coverage']:.0%} of the RUL inputs measured"
        withheld = s["status"] == "insufficient inputs"
        if withheld:
            st.error(f"**Not enough measurements to predict** ({cover}). Fill in more of the table.")
        elif s["status"] == "partial inputs":
            st.warning(f"Partial inputs: {cover}; the rest use training medians.")

        a, b = st.columns(2)
        a.metric("Predicted SoH", "—" if withheld else f"{s['soh_pred']:.1%}",
                 help=f"{model.soh_meta['model']} · typical error {model.soh_meta['loco_MAE']:.1%} on unseen cells")
        b.metric("Measured SoH", "—" if s["soh_measured"] is None else f"{s['soh_measured']:.1%}",
                 help="From the capacity_ah column, when given")
        if not withheld:
            shown_soh = s["soh_measured"] if s["soh_measured"] is not None else s["soh_pred"]
            st.markdown(S.soh_bar(shown_soh, "Measured SoH" if s["soh_measured"] is not None else
                                  "Predicted SoH"), unsafe_allow_html=True)

        at_eol = not withheld and s["rul_note"] == EOL_NOTE
        c, d = st.columns(2)
        rul_help = f"Typical error {s['rul_mae']:.0f} {unit} on unseen cells. {s['rul_note']}"
        c.metric("Estimated RUL", "—" if withheld else "End of life" if at_eol else f"{s['rul']:.0f} {unit}",
                 help=rul_help)
        no_energy = s["energy_retention"] is None or (withheld and s["energy_source"].startswith("proxy"))
        d.metric("Energy retention", "—" if no_energy else f"{s['energy_retention']:.1%}",
                 help=f"Relative to this battery's first rows · {s['energy_source']}")
        if at_eol:
            past = result["eol_at"] if result["eol_at"] is not None else s["ageing"]
            st.caption(f"SoH is at or below 80 % (since {H.AGEING_LABELS[s['ageing_col']].lower()} {past:g}), "
                       "so no useful life is left. Move **Predict as of** back to an earlier point.")
        elif not withheld:
            notes = []
            if result["actual_rul"] is not None:
                notes.append(f"Actual RUL from the full history: **{result['actual_rul']:.0f} {unit}**")
            if s["rul_r2"] < 0:
                notes.append("indicative only (R² < 0 on unseen cells)")
            if notes:
                st.caption(" · ".join(notes))

        if withheld:
            st.caption("Risk and anomalies: not assessed.")
        else:
            st.markdown(H.risk_badge(s["risk_rule"], "Risk: "), unsafe_allow_html=True)
            if not s["anomaly_assessed"]:
                st.caption(f"Anomalies: {s['anomaly_explanation']}")
            elif s["anomaly_latest"]:
                st.error(f"Latest row is anomalous: {s['anomaly_explanation']}")
            else:
                st.success(f"No anomaly on the latest row · {s['anomaly_count']} of {s['rows']} rows "
                           "flagged in the history")

        issues = []
        if s["missing_inputs"]:
            issues.append(f"{len(s['missing_inputs'])} inputs missing")
        if s["out_of_range_latest"]:
            issues.append(f"{len(s['out_of_range_latest'])} values outside the training range")
        if s["quality_flags_latest"]:
            issues.append(s["quality_flags_latest"])
        if issues and not withheld:
            st.warning("Input checks: " + "; ".join(issues) + ".")

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
if result is not None and result["pred"] is not None:
    pred = result["pred"]
    ageing = get_predictor(result["dataset"], result["rul_variant"]).order_col
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    charts = st.container(border=True)
    charts.markdown(S.step(3, "Charts"), unsafe_allow_html=True)
    plotted = pred[pred.prediction_status != "insufficient inputs"]
    if plotted.empty:
        charts.info("Nothing to plot: no row has enough measurements for a prediction.")
    else:
        g1, g2, g3 = charts.columns(3, gap="medium")
        g1.markdown("**State of health**")
        g1.altair_chart(H.soh_chart(plotted, ageing), width="stretch")
        g2.markdown("**Remaining useful life**")
        g2.altair_chart(H.rul_chart(plotted, ageing), width="stretch")
        g3.markdown("**Energy retention**")
        g3.altair_chart(H.energy_chart(plotted, ageing), width="stretch")
    with charts.expander("Results table"):
        st.dataframe(pred, width="stretch", hide_index=True)
        st.download_button("Download results (CSV)", pred.to_csv(index=False),
                           file_name=f"voltara_{result['dataset']}_predictions.csv", mime="text/csv")
