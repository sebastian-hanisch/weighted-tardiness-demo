"""Vehikel A (Neutral) und Vehikel B (Werkstatt/Logistik): Erzeugung, Determinismus; Auswertung: Kennzahlen,
Sweep, Optimalitäts- und Timing-Messreihe, Vehikel-B-Härtetest (Rüstzeiten)."""

from dataclasses import replace

import numpy as np
import pytest

import wt_algorithm as A
import wt_constants as C
import wt_evaluation as ev
import wt_scenario as S
import wt_scenario_logistik as SL


# --- Vehikel A ----------------------------------------------------------------------------------------------------------------------------------


def test_instance_shape_and_bounds():
    inst = S.generate(20, 3)
    assert inst.n == 20 and inst.p.shape == (20,) and inst.d.shape == (20,) and inst.w.shape == (20,)
    assert inst.p.min() >= C.P_MIN and inst.p.max() <= C.P_MAX
    assert (inst.d >= inst.p).all()
    assert set(inst.w.tolist()) <= set(C.WEIGHT_VALUES)


def test_instance_is_deterministic_and_seed_dependent():
    a, b, c = S.generate(30, 5), S.generate(30, 5), S.generate(30, 6)
    assert np.array_equal(a.p, b.p) and np.array_equal(a.d, b.d) and np.array_equal(a.w, b.w)
    assert not np.array_equal(a.p, c.p)


# --- Vehikel B ------------------------------------------------------------------------------------------------------------------------------


def test_logistik_instance_shares_the_same_base_data_as_neutral():
    neutral = S.generate(20, 7)
    logistik = SL.generate(20, 7)
    assert np.array_equal(neutral.p, logistik.p) and np.array_equal(neutral.d, logistik.d) and np.array_equal(neutral.w, logistik.w)


def test_logistik_instance_is_deterministic():
    a, b = SL.generate(10, 2), SL.generate(10, 2)
    assert np.array_equal(a.family, b.family) and np.array_equal(a.setup, b.setup)


# --- Analyse --------------------------------------------------------------------------------------------------------------------------------


def test_analysis_fields_are_consistent():
    a = ev.analyse(ev.Settings(n=20))
    assert a.optimal is None                                    # n=20 > EXACT_MAX_N


def test_analysis_matches_the_optimum_for_a_small_easy_instance():
    """ATC ist keine bewiesene Regel (anders als Stück 1-4) - hier nur geprüft, dass CP-SAT für kleines n
    überhaupt bewiesen löst, NICHT dass ATC immer trifft (siehe test_claims.py für die ehrliche Trefferquote)."""
    a = ev.analyse(ev.Settings(n=6))
    assert a.optimal is not None and a.optimal_proven


# --- Vehikel-Bewusstsein der Hauptanalyse (von Anfang an, siehe [[feedback_vehicle_toggle_must_drive_primary_metrics]]) ----------------------


def test_analyse_on_the_logistik_vehicle_actually_uses_setup_aware_completion_times():
    settings = ev.Settings(n=8, seed=100000, vehicle="logistik", setup_time=30, n_families=3)
    a = ev.analyse(settings)
    linst = ev.logistik_instance(8, 100000, 3, 30)
    independent_atc = A.evaluate_order(linst.p, linst.d, linst.w, a.atc.order, linst.family, linst.setup)
    assert a.atc.total == pytest.approx(independent_atc.total)
    assert not np.array_equal(a.atc.completion, np.cumsum(linst.p[a.atc.order]))  # Rüstzeiten verschieben die Fertigstellung


def test_analyse_on_the_neutral_vehicle_is_unaffected_by_logistik_only_settings():
    a1 = ev.analyse(ev.Settings(n=10, seed=5, vehicle="neutral", setup_time=5))
    a2 = ev.analyse(ev.Settings(n=10, seed=5, vehicle="neutral", setup_time=60))
    assert a1.atc.total == pytest.approx(a2.atc.total)


def test_switching_vehicle_actually_changes_the_atc_total():
    a_neutral = ev.analyse(ev.Settings(n=10, seed=7, vehicle="neutral"))
    a_logistik = ev.analyse(ev.Settings(n=10, seed=7, vehicle="logistik", setup_time=60, n_families=2))
    assert a_neutral.atc.total != pytest.approx(a_logistik.atc.total)


def test_analysis_is_deterministic_given_the_chain_seed():
    s = ev.Settings(n=20, seed=1, chain_seed=0)
    a, b, c = ev.analyse(s), ev.analyse(s), ev.analyse(replace(s, chain_seed=1))
    assert a.gap_random == pytest.approx(b.gap_random)
    assert a.gap_random != pytest.approx(c.gap_random)


# --- Sweep und Messreihe -------------------------------------------------------------------------------------------------------------------


def test_run_config_counts_runs_and_aggregates():
    r = ev.run_config(ev.Settings(n=15))
    assert r["n_runs"] == len(C.SWEEP_SEEDS) * C.SWEEP_CHAINS


def test_sweep_values_labels_and_ordering():
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    rows = ev.sweep("n", ev.Settings(), (5, 40))
    assert [r["value"] for r in rows] == [5, 40]


def test_optimality_check_reports_a_proven_rate():
    rows = ev.optimality_check(ns=(3, 4, 5), seeds=C.SWEEP_SEEDS[:2])
    assert all(0.0 <= r["match_rate"] <= 1.0 and r["proven_rate"] == 1.0 for r in rows)


def test_timing_sweep_shows_exact_growing_far_slower_at_small_n_than_atc():
    rows = ev.timing_sweep(ns=(3, 8))
    small, large = rows[0], rows[1]
    assert large["exact_seconds"] > small["exact_seconds"]
    assert large["atc_seconds"] < 0.01


def test_setup_gap_grows_with_the_setup_time():
    small = ev.setup_gap(n=7, setup_time=0)
    large = ev.setup_gap(n=7, setup_time=60)
    assert large["gap_mean"] >= small["gap_mean"]
