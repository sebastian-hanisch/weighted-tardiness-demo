"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt, Randwerte, Würfel-Knöpfe, Permalink-Grenzen,
Vehikel-Umschalter, Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import wt_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(wt_step=1, **state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if wt_step != 1:
        at.select_slider(key="wt_step").set_value(wt_step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_the_measured_default():
    at = _run()
    _ok(at)
    assert _metric(at, "ATC (Σ wⱼTⱼ)") == "539"
    assert _metric(at, "EDD (ignoriert Gewichte)") == "+51.4 %"
    assert _metric(at, "WSPT (ignoriert Fristen)") == "+262.0 %"
    assert _metric(at, "Zufällige Reihenfolge (Mittel über 20)") == "+1222.0 %"


def test_switching_to_the_logistik_vehicle_actually_changes_the_main_metric():
    """Regressionsschutz für dieselbe Lücke, die in Stück 1-3 dieser Linie gefunden wurde: der Vehikel-
    Umschalter muss die HAUPT-Kennzahl ändern, nicht nur eine separate Box."""
    at_neutral = _run(n_slider=10, seed_input=7, vehicle_radio="neutral")
    at_logistik = _run(n_slider=10, seed_input=7, vehicle_radio="logistik", setup_time_slider=60, n_families_slider=2)
    _ok(at_neutral)
    _ok(at_logistik)
    assert _metric(at_neutral, "ATC (Σ wⱼTⱼ)") != _metric(at_logistik, "ATC (Σ wⱼTⱼ)")


def test_negative_gap_is_shown_honestly_not_as_a_broken_sign():
    """Regressionsschutz für dieselbe Lücke, die in Stück 4 (wspt-demo) gefunden wurde: ATC ist keine bewiesene
    Regel, WSPT kann hier (n=12, Seed 1, Rüstzeit 15) sogar besser abschneiden. Die Anzeige darf weder
    '+-9.4 %' zeigen noch die Warnung unterschlagen."""
    at = _run(n_slider=12, seed_input=1, vehicle_radio="logistik", setup_time_slider=15, n_families_slider=3)
    _ok(at)
    wspt_metric = _metric(at, "WSPT (ignoriert Fristen)")
    assert "+-" not in wspt_metric and wspt_metric.startswith("-")
    assert any("schneidet ATC hier sogar schlechter ab" in w.value for w in at.warning)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["n_slider"] == p["n"] and at.session_state["vehicle_radio"] == p["vehicle"]
    assert at.metric


@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.parametrize("vehicle", ["neutral", "logistik"])
def test_every_step_runs_on_both_vehicles(step, vehicle):
    at = _run(n_slider=10, vehicle_radio=vehicle, wt_step=step)
    _ok(at)
    assert at.session_state["wt_step"] == step


def test_step_two_has_the_eingeplant_slider_defaulting_to_all_jobs():
    at = _run(n_slider=12, wt_step=2)
    _ok(at)
    lv = next(s for s in at.slider if s.key == "wt_upto")
    assert lv.value == lv.max == 12


def test_exact_limit_is_respected_in_the_metric():
    """Bei n = EXACT_MAX_N kann CP-SAT entweder beweisen (Label 'CP-SAT (exakte Gegenprobe)') oder - auf
    langsamerer Hardware - das Zeitlimit erreichen (Label 'CP-SAT', Wert 'Zeitlimit erreicht'); beides ist ein
    gültiger, in app.py behandelter Zustand, kein Fehler."""
    at = _run(n_slider=C.EXACT_MAX_N)
    _ok(at)
    proven_metric = next((m for m in at.metric if m.label == "CP-SAT (exakte Gegenprobe)"), None)
    timeout_metric = next((m for m in at.metric if m.label == "CP-SAT" and m.value == "Zeitlimit erreicht"), None)
    assert (proven_metric is not None and proven_metric.value in ("trifft ATC exakt", "ATC weicht ab")) or timeout_metric is not None
    at2 = _run(n_slider=C.EXACT_MAX_N + 1)
    _ok(at2)
    assert "erst ab n" in _metric(at2, "CP-SAT")


def test_dice_buttons_change_the_seeds():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old
    old_c = at.session_state["chain_seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Kette würfeln").click().run()
    _ok(at)
    assert at.session_state["chain_seed_input"] != old_c


@pytest.mark.parametrize("kw", [dict(n_slider=C.N_MIN), dict(vehicle_radio="logistik", setup_time_slider=C.SETUP_TIME_MIN),
                                 dict(vehicle_radio="logistik", setup_time_slider=C.SETUP_TIME_MAX), dict(vehicle_radio="logistik", n_families_slider=C.N_FAMILIES_MIN)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["vehicle"] = "logistik"
    at.query_params["setup"] = "9999"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["vehicle_radio"] == "logistik"
    assert at.session_state["setup_time_slider"] == C.SETUP_TIME_MAX


def test_permalink_ignores_an_invalid_vehicle():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["vehicle"] = "nicht_vorhanden"
    at.run()
    _ok(at)
    assert at.session_state["vehicle_radio"] == C.DEFAULT_VEHICLE


def test_sweeps_run_on_demand():
    at = _run(n_slider=15)
    at.selectbox(key="sweep_select").set_value("n").run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_experiments_run_on_demand():
    at = _run(n_slider=7, vehicle_radio="logistik")
    for key, flag in (("opt_start", "opt_on"), ("timing_start", "timing_on"), ("setup_start", "setup_on")):
        next(b for b in at.button if b.key == key).click().run()
        _ok(at)
        assert at.session_state[flag]


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Es gibt eine einfache, beweisbar optimale Regel" in m.value for m in at.markdown)
