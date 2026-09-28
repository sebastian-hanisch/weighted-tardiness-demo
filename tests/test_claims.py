"""Jede Zahl der App-Texte ist hier über die fünf festen Sweep-Instanzen (je drei Ketten) belegt. Positive UND
negative Aussagen: ATC schlägt EDD/WSPT/Zufall deutlich im Standardfall - UND trifft NICHT zuverlässig das
CP-SAT-Optimum (anders als die bewiesenen Regeln in Stück 1-4 dieser Linie). Rechenzeiten nur als
Größenordnung geprüft; teure Läufe (CP-SAT) sind modul-weit über lru_cache dedupliziert."""

from functools import lru_cache

import pytest

import wt_evaluation as ev


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


@lru_cache(maxsize=1)
def _opt():
    return tuple(tuple(r.items()) for r in ev.optimality_check())


def opt_rows():
    return [dict(r) for r in _opt()]


@lru_cache(maxsize=1)
def _timing():
    return tuple(tuple(r.items()) for r in ev.timing_sweep())


def timing_rows():
    return [dict(r) for r in _timing()]


@lru_cache(maxsize=1)
def _setup_sweep():
    return tuple(tuple(r.items()) for r in ev.setup_gap_sweep())


def setup_rows():
    return [dict(r) for r in _setup_sweep()]


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Standardfall -------------------------------------------------------------------------------------------------------------------------------


def test_standard_case_numbers():
    std = cfg()
    near(std["gap_edd"], 111.6, 40.0)
    near(std["gap_wspt"], 189.6, 60.0)
    near(std["gap_random"], 597.0, 200.0)


def test_atc_beats_edd_wspt_and_random_on_average_at_the_standard_size():
    for n in (10, 20, 40):
        row = cfg(n=n)
        assert row["gap_edd"] > 0.0 and row["gap_wspt"] > 0.0 and row["gap_random"] > 0.0


# --- Optimalität gegen CP-SAT: KEIN Beweis-Check, ehrliche Trefferquote ------------------------------------------------------------------------


def test_cp_sat_proves_most_small_instances():
    """Schwelle bewusst nicht zu scharf (0.6 statt z. B. 0.9): auf einer kernärmeren Maschine (z. B. ein
    CI-Runner) braucht CP-SAT bei n=9 im Einzelfall länger, ohne dass das ein echter Regressionsfehler wäre -
    siehe [[feedback_ci_platform_robust_tests]]."""
    rows = opt_rows()
    assert all(r["proven_rate"] >= 0.6 for r in rows)


def test_atc_does_not_reliably_match_the_true_optimum():
    """Die zentrale, ehrliche Aussage dieses Stücks: ANDERS als SPT/EDD/Moore-Hodgson/WSPT (100 % Trefferquote
    in Stück 1-4) trifft ATC das CP-SAT-Optimum NICHT zuverlässig - es ist eine Heuristik, kein Beweis."""
    rows = opt_rows()
    assert any(r["match_rate"] < 1.0 for r in rows)


# --- Timing: CP-SAT (exponentiell im schlimmsten Fall) gegen ATC (O(n²)) -----------------------------------------------------------------------


def test_exact_solving_grows_far_slower_at_small_n_than_at_the_exact_limit():
    rows = timing_rows()
    small = next(r for r in rows if r["value"] == 2)
    large = next(r for r in rows if r["value"] == 9)
    assert large["exact_seconds"] > small["exact_seconds"] * 10
    assert large["atc_seconds"] < 0.01


# --- Vehikel B: Rüstzeit-Härtetest ------------------------------------------------------------------------------------------------------------


def test_setup_gap_is_small_but_not_necessarily_zero_without_setup_time():
    """Anders als Stück 1-4: der Rüstzeit-0-Fall ist KEIN garantierter Nulltreffer, weil ATC schon ohne
    Rüstzeiten keine bewiesene Regel ist (siehe test_atc_does_not_reliably_match_the_true_optimum)."""
    row = setup_rows()[0]
    assert row["value"] == 0
    assert 0.0 <= row["gap_mean"] < 15.0


def test_setup_gap_trends_upward_with_the_setup_time():
    rows = setup_rows()
    assert rows[-1]["gap_mean"] > rows[0]["gap_mean"]
