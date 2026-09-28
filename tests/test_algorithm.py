"""wt_algorithm: ATC-Verhalten, EDD/WSPT als falsche Regeln, CP-SAT gegen unabhängige Brute-Force-
Vollaufzählung (nur für kleine n - CP-SAT selbst wird hier geprüft, nicht ATC, das ist KEIN Beweis-Check),
Rüstzeit-Variante, Handrechnung."""

import itertools

import numpy as np
import pytest

import wt_algorithm as A


def _pdw(seed, n):
    rng = np.random.default_rng(seed)
    p = rng.integers(1, 20, size=n).astype(np.int64)
    total = int(p.sum())
    d = rng.integers(1, max(total, 2), size=n).astype(np.int64)
    w = rng.choice((1, 2, 4), size=n).astype(np.int64)
    return p, d, w


def _brute_force(p, d, w):
    n = len(p)
    best = None
    for perm in itertools.permutations(range(n)):
        total = A.evaluate_order(p, d, w, np.array(perm)).total
        if best is None or total < best:
            best = total
    return best


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
def test_cp_sat_matches_brute_force_for_every_seed(n):
    """CP-SAT ist der exakte Löser dieses Stücks (ersetzt die reine Brute-Force-Gegenprobe der Stücke 1-4) -
    hier gegen eine UNABHÄNGIGE Brute-Force-Vollaufzählung geprüft, damit ein Fehler im Kreis-Modell nicht
    unbemerkt bliebe."""
    for seed in range(5):
        p, d, w = _pdw(seed, n)
        bf = _brute_force(p, d, w)
        result, proven = A.solve_exact(p, d, w, time_limit_seconds=10)
        assert proven
        assert result.total == pytest.approx(bf, abs=1e-6)


def test_completion_and_tardiness_are_computed_correctly():
    p = np.array([5, 2, 8, 1])
    d = np.array([3, 10, 20, 1])
    w = np.array([1, 1, 1, 1])
    order = np.array([3, 1, 0, 2])          # p sortiert: 1, 2, 5, 8 -> completion 1, 3, 8, 16
    result = A.evaluate_order(p, d, w, order)
    assert result.completion.tolist() == [1, 3, 8, 16]
    assert result.tardiness.tolist() == [0, 0, 5, 0]        # nur Auftrag 0 (completion 8, due 3) ist verspätet
    assert result.total == pytest.approx(5.0)


def test_edd_order_is_ascending_by_due_date():
    d = np.array([5, 2, 8, 1, 2])
    order = A.edd_order(d)
    assert d[order].tolist() == sorted(d.tolist())


def test_wspt_order_is_descending_by_weight_over_processing_time():
    p = np.array([5, 2, 8, 1, 2])
    w = np.array([1, 4, 1, 1, 2])
    order = A.wspt_order(p, w)
    ratios = (w / p)[order]
    assert list(ratios) == sorted(ratios, reverse=True)


def test_atc_beats_edd_and_wspt_on_a_hand_picked_instance():
    """Handrechnung: ein wichtiger Auftrag mit knapper Frist und ein unwichtiger mit lockerer Frist - EDD
    (ignoriert Gewicht) und WSPT (ignoriert Frist) treffen beide die falsche Entscheidung."""
    p = np.array([4, 4])
    d = np.array([4, 100])           # Auftrag 0 hat eine knappe Frist
    w = np.array([10, 1])            # Auftrag 0 ist zehnmal so wichtig
    atc_result = A.atc(p, d, w)
    assert atc_result.order.tolist() == [0, 1]         # der wichtige, fristnahe Auftrag zuerst
    assert atc_result.total == pytest.approx(0.0)       # beide pünktlich in dieser Reihenfolge


def test_atc_order_is_deterministic():
    p, d, w = _pdw(3, 12)
    a = A.atc_order(p, d, w)
    b = A.atc_order(p, d, w)
    assert a.tolist() == b.tolist()


def test_random_order_is_deterministic_given_the_rng_state():
    n = 8
    a = A.random_order(n, np.random.default_rng(0))
    b = A.random_order(n, np.random.default_rng(0))
    assert a.tolist() == b.tolist()
    assert sorted(a.tolist()) == list(range(n))


# --- Mit Rüstzeiten (Vehikel B) --------------------------------------------------------------------------------


def test_atc_order_at_zero_setup_time_matches_the_neutral_vehicle_exactly():
    """Die richtige Konsistenzprüfung für dieses Stück: ANDERS als bei Stück 1-4 (wo die Regel selbst bewiesen
    optimal ist und deshalb bei Rüstzeit 0 automatisch das Optimum trifft) ist ATC keine bewiesene Regel - der
    Kollaps-Test prüft deshalb NICHT 'trifft ATC das Optimum', sondern nur, dass Vehikel B bei Rüstzeit 0
    strukturell exakt auf Vehikel A zurückfällt (derselbe Auftrags-Index, dieselbe Reihenfolge)."""
    p, d, w = _pdw(7, 10)
    family = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    setup = np.zeros((2, 2))
    neutral_order = A.atc_order(p, d, w)
    logistik_order = A.atc_order(p, d, w, family, setup)
    assert neutral_order.tolist() == logistik_order.tolist()


def test_setup_time_is_only_charged_on_a_family_change():
    p = np.array([2, 2, 2])
    d = np.array([100, 100, 100])
    w = np.array([1, 1, 1])
    family = np.array([0, 0, 1])
    setup = np.array([[0, 10], [10, 0]])
    order = np.array([0, 1, 2])
    result = A.evaluate_order(p, d, w, order, family, setup)
    assert result.completion.tolist() == [2, 4, 4 + 10 + 2]


def test_cp_sat_with_setup_matches_independent_brute_force():
    p, d, w = _pdw(11, 6)
    rng = np.random.default_rng(11)
    family = rng.integers(0, 3, size=6)
    setup = np.array([[0, 5, 8], [5, 0, 3], [8, 3, 0]])

    def bf_setup():
        best = None
        for perm in itertools.permutations(range(6)):
            total = A.evaluate_order(p, d, w, np.array(perm), family, setup).total
            if best is None or total < best:
                best = total
        return best

    bf = bf_setup()
    result, proven = A.solve_exact(p, d, w, family, setup, time_limit_seconds=10)
    assert proven
    assert result.total == pytest.approx(bf, abs=1e-6)


def test_cp_sat_setup_matches_no_setup_when_setup_is_zero():
    p, d, w = _pdw(13, 6)
    family = np.array([0, 1, 0, 1, 0, 1])
    setup0 = np.zeros((2, 2))
    plain, proven1 = A.solve_exact(p, d, w, time_limit_seconds=10)
    with_setup, proven2 = A.solve_exact(p, d, w, family, setup0, time_limit_seconds=10)
    assert proven1 and proven2
    assert with_setup.total == pytest.approx(plain.total, abs=1e-6)
