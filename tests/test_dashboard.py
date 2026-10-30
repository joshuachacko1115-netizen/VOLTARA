"""Dashboard tests: the UI must show exactly what the inference module predicts.

AppTest runs dashboard/app.py headlessly (no browser), so these cover the real
widgets, session state and error handling.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR, PROCESSED_DIR

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from dashboard import helpers as H  # noqa: E402
from src.inference.predict import VoltaraPredictor, example_history  # noqa: E402
from src.models.range_estimation import RangeAssumptions  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "anomaly_ba.joblib").exists() or not (PROCESSED_DIR / "ba_cycles.parquet").exists(),
    reason="trained models / processed data not available - run `python -m src.run_pipeline`")

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120).run()


def _no_exceptions(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def _as_of_default(dataset: str, hist: pd.DataFrame) -> pd.DataFrame:
    """The history up to the dashboard's default "Predict as of" row."""
    cut = H.default_cutoff(VoltaraPredictor(dataset).predict(hist))
    return hist.iloc[:cut + 1].reset_index(drop=True)


@pytest.mark.parametrize("dataset,cell", [("battery_aging", "B0005"), ("randomized_usage", "RW9")])
def test_dashboard_matches_the_inference_module(dataset, cell):
    at = _app()
    at.selectbox[0].select(dataset).run()
    at.selectbox[1].select(cell).run()
    at.button[0].click().run()
    _no_exceptions(at)
    shown = at.session_state["result"]["pred"]
    direct = VoltaraPredictor(dataset).predict(_as_of_default(dataset, example_history(dataset, cell)))
    pd.testing.assert_frame_equal(shown, direct)
    last = direct.iloc[-1]
    assert _metric(at, "Predicted SoH") == f"{last.soh_pred:.1%}"
    assert _metric(at, "Estimated RUL") == f"{last.rul_pred:.0f} {last.rul_unit}"
    assert last.rul_pred > 0
    assert any(f"Risk: {last.risk_rule}" in m.value for m in at.markdown)
    assert any("Actual RUL from the full history" in c.value for c in at.caption)


def test_default_point_is_before_end_of_life_and_full_history_says_end_of_life():
    full = example_history("battery_aging", "B0005")
    whole = VoltaraPredictor("battery_aging").predict(full)
    eol = H.end_of_life_index(whole)
    assert eol is not None and 0 < H.default_cutoff(whole) < eol
    at = _app()
    at.select_slider[0].set_value(len(full) - 1).run()
    at.button[0].click().run()
    _no_exceptions(at)
    assert _metric(at, "Estimated RUL") == "End of life"
    assert any("no useful life is left" in c.value for c in at.caption)


def test_examples_skip_cells_that_start_below_end_of_life():
    at = _app()
    shown = at.selectbox[1].options
    assert "B0005" in shown and "B0029" in shown
    assert "B0033" not in shown and "B0034" not in shown


def test_blank_template_runs_and_reports_missing_measurements():
    at = _app()
    at.radio[0].set_value("Blank template").run()
    at.button[0].click().run()
    _no_exceptions(at)
    pred = at.session_state["result"]["pred"]
    assert (pred.soh_source == "predicted").all()
    assert len(pred.missing_inputs.iloc[0].split(", ")) == len(VoltaraPredictor("battery_aging").required_inputs()) - 1
    assert _metric(at, "Measured SoH") == "—"
    # Nothing measured: the numbers would only describe the training medians, so they are withheld.
    assert (pred.prediction_status == "insufficient inputs").all()
    assert _metric(at, "Predicted SoH") == "—" and _metric(at, "Estimated RUL") == "—"
    assert any("Not enough measurements" in e.value for e in at.error)
    assert not any("Risk: " in m.value for m in at.markdown)
    assert not any("end of life" in c.value for c in at.caption)
    assert any("Nothing to plot" in i.value for i in at.info)
    assert not any("no anomaly" in s.value.lower() for s in at.success)


def test_uploaded_csv_gives_the_same_result_as_the_module():
    hist = example_history("randomized_usage", "RW3")
    at = _app()
    at.selectbox[0].select("randomized_usage").run()
    at.radio[0].set_value("Upload CSV").run()
    at.file_uploader[0].upload("rw3.csv", hist.to_csv(index=False).encode(), "text/csv").run()
    at.button[0].click().run()
    _no_exceptions(at)
    direct = VoltaraPredictor("randomized_usage").predict(_as_of_default("randomized_usage", hist))
    pd.testing.assert_frame_equal(at.session_state["result"]["pred"], direct)


def test_bad_csv_shows_an_error_not_a_crash():
    at = _app()
    at.radio[0].set_value("Upload CSV").run()
    at.file_uploader[0].upload("bad.csv", b"cycle_number,chg_ah\n1,abc\n2,1.8\n", "text/csv").run()
    at.button[0].click().run()
    _no_exceptions(at)
    assert at.session_state["result"]["pred"] is None
    assert any("non-numeric" in e.value and "'chg_ah'" in e.value and "'abc'" in e.value for e in at.error)


def test_censoring_aware_variant_from_the_ui():
    at = _app()
    at.selectbox[0].select("randomized_usage").run()
    at.radio[1].set_value("censoring_aware").run()
    at.button[0].click().run()
    _no_exceptions(at)
    r = at.session_state["result"]
    assert r["rul_variant"] == "censoring_aware"
    assert "never-failed" in r["pred"].rul_note.iloc[0]
    assert not any("km" in m.label for m in at.metric)


def test_changed_inputs_mark_the_result_as_out_of_date():
    at = _app()
    at.button[0].click().run()
    at.selectbox[1].select("B0006").run()
    assert any("inputs changed" in w.value for w in at.warning)


# --------------------------------------------------------------------------- #
# helpers (no UI)
# --------------------------------------------------------------------------- #
def test_unfamiliar_values_are_flagged():
    p = VoltaraPredictor("battery_aging")
    hist = example_history("battery_aging", "B0005")
    hist.loc[hist.index[-1], "chg_t_mean"] = 80.0
    pred, err = H.predict_or_error(p, hist, None)
    assert err is None
    assert "chg_t_mean" in H.latest_summary(pred, p)["out_of_range_latest"]


@pytest.mark.parametrize("bad,msg", [
    (pd.DataFrame(), "empty"),
    (pd.DataFrame({"chg_ah": [1.0]}), "ageing column"),
    (pd.DataFrame({"cycle_number": [1, None]}), "filled in"),
    (pd.DataFrame({"cycle_number": [-1]}), "negative"),
])
def test_prediction_errors_become_messages(bad, msg):
    pred, err = H.predict_or_error(VoltaraPredictor("battery_aging"), bad, None)
    assert pred is None and msg in err


def test_read_history_csv_rejects_garbage():
    with pytest.raises(ValueError):
        H.read_history_csv(b"")
    with pytest.raises(ValueError):
        H.read_history_csv(b"cycle_number\n")


@pytest.mark.parametrize("dataset,cell,drop", [("battery_aging", "B0005", []),
                                               ("randomized_usage", "RW9", ["capacity_ah", "energy_wh"])])
def test_charts_build(dataset, cell, drop):
    p = VoltaraPredictor(dataset)
    pred = p.predict(example_history(dataset, cell).drop(columns=drop),
                     range_assumptions=RangeAssumptions())
    for chart in (H.soh_chart(pred, p.order_col), H.rul_chart(pred, p.order_col),
                  H.energy_chart(pred, p.order_col)):
        assert chart.to_dict()


def test_fingerprint_changes_with_any_input():
    hist = example_history("battery_aging", "B0005")
    f = H.fingerprint("battery_aging", hist, "default", None)
    assert f == H.fingerprint("battery_aging", hist.copy(), "default", None)
    hist2 = hist.copy()
    hist2.iloc[0, 1] = np.nan
    assert f != H.fingerprint("battery_aging", hist2, "default", None)
    assert f != H.fingerprint("battery_aging", hist, "censoring_aware", None)
    assert f != H.fingerprint("battery_aging", hist, "default", RangeAssumptions())


def test_history_cut_off_predicts_as_of_that_point():
    full = example_history("battery_aging", "B0006")
    at = _app()
    at.selectbox[1].select("B0006").run()
    at.select_slider[0].set_value(39).run()
    at.button[0].click().run()
    _no_exceptions(at)
    shown = at.session_state["result"]["pred"]
    assert len(shown) == 40
    p = VoltaraPredictor("battery_aging")
    pd.testing.assert_frame_equal(shown, p.predict(full.head(40)))
    # causal: the same rows inside the full-history prediction get the same RUL
    np.testing.assert_allclose(shown.rul_pred, p.predict(full).rul_pred.iloc[:40])


def test_partial_inputs_warn_but_still_predict():
    hist = example_history("battery_aging", "B0005").drop(columns=["re_ohm", "rct_ohm", "chg_t_mean"])
    at = _app()
    at.radio[0].set_value("Upload CSV").run()
    at.file_uploader[0].upload("partial.csv", hist.to_csv(index=False).encode(), "text/csv").run()
    at.button[0].click().run()
    _no_exceptions(at)
    pred = at.session_state["result"]["pred"]
    assert pred.prediction_status.iloc[-1] == "partial inputs"
    assert any("Partial inputs" in w.value for w in at.warning)
    assert _metric(at, "Predicted SoH") != "—"


# --------------------------------------------------------------------------- #
# real-world CSV files
# --------------------------------------------------------------------------- #
def _ba():
    return example_history("battery_aging", "B0006")


@pytest.mark.parametrize("name,data", [
    ("semicolon", lambda d: d.to_csv(index=False, sep=";").encode()),
    ("semicolon + decimal comma", lambda d: d.to_csv(index=False, sep=";", decimal=",").encode()),
    ("tab", lambda d: d.to_csv(index=False, sep="\t").encode()),
    ("UTF-8 BOM", lambda d: b"\xef\xbb\xbf" + d.to_csv(index=False).encode()),
    ("Windows cp1252", lambda d: d.to_csv(index=False).encode("cp1252")),
    ("header case and spaces", lambda d: d.rename(columns={"cycle_number": " Cycle Number ",
                                                            "chg_ah": "CHG-AH"}).to_csv(index=False).encode()),
])
def test_csv_variants_give_the_same_prediction(name, data):
    hist = _ba()
    parsed = H.read_history_csv(data(hist))
    p = VoltaraPredictor("battery_aging")
    pd.testing.assert_frame_equal(p.predict(parsed), p.predict(hist), check_exact=False, rtol=1e-12)


@pytest.mark.parametrize("change,msg", [
    (lambda d: d.assign(chg_ah=np.inf), "infinite or implausibly large"),
    (lambda d: d.assign(chg_ah=1e300), "infinite or implausibly large"),
])
def test_impossible_numbers_are_rejected_clearly(change, msg):
    _, err = H.predict_or_error(VoltaraPredictor("battery_aging"), change(_ba()), None)
    assert msg in err and "chg_ah" in err


def test_wrong_dataset_file_says_which_dataset_to_pick():
    rw = example_history("randomized_usage", "RW5")
    _, err = H.predict_or_error(VoltaraPredictor("battery_aging"), rw, None)
    assert "Randomized Usage" in err and "switch" in err


def test_missing_value_tokens_and_duplicates():
    hist = _ba().astype(object)
    hist["re_ohm"] = "-"
    hist.loc[1, "chg_ah"] = "n/a"
    hist = pd.concat([hist, hist.tail(1)], ignore_index=True)
    pred, err = H.predict_or_error(VoltaraPredictor("battery_aging"), hist, None)
    assert err is None
    assert "re_ohm" in pred.missing_inputs.iloc[0]
    assert pred.data_quality_flags.iloc[-1].count("same cycle_number") == 1


def test_binary_and_empty_files_are_rejected():
    for bad in (bytes(range(256)) * 20, b"", b"   \n  "):
        with pytest.raises(ValueError):
            H.read_history_csv(bad)
