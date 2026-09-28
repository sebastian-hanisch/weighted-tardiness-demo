"""Auswertung der Gewichtete-Verspätung-Demo: ATC gegen EDD (ignoriert Gewichte), WSPT (ignoriert Fristen) und
zufällige Reihenfolgen, gegen CP-SAT als exakte Gegenprobe (nur kleine n, stark NP-schwer), und das
Vehikel-B-Experiment (bleibt ATC nahe am Optimum, sobald Rüstzeiten dazukommen)."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import wt_algorithm as A
import wt_constants as C
import wt_scenario as S
import wt_scenario_logistik as SL


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    chain_seed: int = 0
    tf: float = C.DEFAULT_TF
    rdd: float = C.DEFAULT_RDD
    vehicle: str = C.DEFAULT_VEHICLE
    setup_time: int = C.DEFAULT_SETUP_TIME
    n_families: int = C.DEFAULT_N_FAMILIES


@lru_cache(maxsize=512)
def instance(n, seed, tf=C.DEFAULT_TF, rdd=C.DEFAULT_RDD):
    return S.generate(n, seed, tf=tf, rdd=rdd)


@lru_cache(maxsize=512)
def logistik_instance(n, seed, n_families, setup_time, tf=C.DEFAULT_TF, rdd=C.DEFAULT_RDD):
    return SL.generate(n, seed, n_families=n_families, setup_time=setup_time, tf=tf, rdd=rdd)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    atc: object
    edd: object              # falsche Regel: ignoriert Gewichte
    wspt: object              # falsche Regel: ignoriert Fristen
    random_mean: float
    random_runs: int
    optimal: object           # None, wenn n > EXACT_MAX_N
    optimal_proven: bool      # False, wenn CP-SAT das Zeitlimit erreicht hat (Ergebnis nicht notwendig optimal)

    @property
    def gap_edd(self):
        return _gap(self.edd.total, self.atc.total)

    @property
    def gap_wspt(self):
        return _gap(self.wspt.total, self.atc.total)

    @property
    def gap_random(self):
        return _gap(self.random_mean, self.atc.total)

    @property
    def atc_matches_optimum(self):
        return self.optimal is not None and abs(self.atc.total - self.optimal.total) < 1e-6


def _gap(value, atc_total):
    """Prozent-Abstand einer Vergleichsregel zu ATC. `atc_total` kann 0 sein (keine Verspätung überhaupt) -
    dann ist jeder positive Abstand per Definition unendlich groß; wir begrenzen ihn auf den absoluten
    Unterschied, damit die App keine Inf/NaN-Anzeige bekommt."""
    if atc_total <= 1e-9:
        return 0.0 if value <= 1e-9 else float(value)
    return 100.0 * (value - atc_total) / atc_total


def analyse(settings, random_draws=20):
    """Wertet ATC auf dem gewählten Vehikel aus - Neutral oder Werkstatt/Logistik (Rüstzeit beim
    Familienwechsel zählt mit, inklusive des zweiten ATC-Terms). Die BEWERTUNG jeder Reihenfolge (und damit
    auch von CP-SAT) wechselt mit dem Vehikel, damit die Haupt-Kennzahlen ehrlich widerspiegeln, was auf dem
    gewählten Vehikel tatsächlich passiert - von Anfang an vehikel-bewusst gebaut (Lehre aus Stück 1-4 dieser
    Linie, siehe [[feedback_vehicle_toggle_must_drive_primary_metrics]])."""
    if settings.vehicle == "logistik":
        inst = logistik_instance(settings.n, settings.seed, settings.n_families, settings.setup_time, settings.tf, settings.rdd)
        p, d, w, family, setup = inst.p, inst.d, inst.w, inst.family, inst.setup

        def ev(order):
            return A.evaluate_order(p, d, w, order, family, setup)

        atc_order = A.atc_order(p, d, w, family, setup)
        if settings.n <= C.EXACT_MAX_N:
            optimal, proven = A.solve_exact(p, d, w, family, setup, C.EXACT_TIME_LIMIT_SECONDS)
        else:
            optimal, proven = None, False
    else:
        inst = instance(settings.n, settings.seed, settings.tf, settings.rdd)
        p, d, w = inst.p, inst.d, inst.w

        def ev(order):
            return A.evaluate_order(p, d, w, order)

        atc_order = A.atc_order(p, d, w)
        if settings.n <= C.EXACT_MAX_N:
            optimal, proven = A.solve_exact(p, d, w, time_limit_seconds=C.EXACT_TIME_LIMIT_SECONDS)
        else:
            optimal, proven = None, False

    atc = ev(atc_order)
    edd = ev(A.edd_order(d))
    wspt = ev(A.wspt_order(p, w))
    rng = np.random.default_rng(settings.chain_seed)
    random_totals = [ev(A.random_order(settings.n, rng)).total for _ in range(random_draws)]
    return Analysis(settings, inst, atc, edd, wspt, float(np.mean(random_totals)), random_draws, optimal, proven)


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS, **changes):
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        for ch in range(chains):
            a = analyse(replace(s0, seed=seed, chain_seed=ch))
            rows.append({"gap_edd": a.gap_edd, "gap_wspt": a.gap_wspt, "gap_random": a.gap_random})
    out = {k: _mean(rows, k) for k in rows[0]}
    out["n_runs"] = len(rows)
    return out


SWEEP_VALUES = {"n": (2, 5, 10, 20, 40, 60)}
SWEEP_LABELS = {"n": "Aufträge"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def optimality_check(ns=C.EXACT_SWEEP_N, seeds=C.SWEEP_SEEDS):
    """ATC gegen CP-SAT über mehrere n und Instanzen - Anteil exakter Treffer UND Anteil bewiesener Läufe
    (CP-SAT kann bei größerem n ans Zeitlimit stoßen, dann ist das Ergebnis nur die beste GEFUNDENE Lösung)."""
    rows = []
    for n in ns:
        matches, proven_count = 0, 0
        for seed in seeds:
            inst = instance(n, seed)
            atc_total = A.atc(inst.p, inst.d, inst.w).total
            opt, proven = A.solve_exact(inst.p, inst.d, inst.w, time_limit_seconds=C.EXACT_TIME_LIMIT_SECONDS)
            if proven:
                proven_count += 1
                if abs(atc_total - opt.total) < 1e-6:
                    matches += 1
        rows.append({"value": n, "match_rate": matches / max(proven_count, 1), "proven_rate": proven_count / len(seeds)})
    return rows


def timing_sweep(ns=C.EXACT_SWEEP_N, seed=C.DEFAULT_SEED):
    """Gemessene Rechenzeit: CP-SAT (Zeitlimit, exponentiell im schlimmsten Fall) gegen ATC (O(n²))."""
    rows = []
    for n in ns:
        inst = instance(n, seed)
        t0 = time.perf_counter()
        A.solve_exact(inst.p, inst.d, inst.w, time_limit_seconds=C.EXACT_TIME_LIMIT_SECONDS)
        t_exact = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(20):
            A.atc(inst.p, inst.d, inst.w)
        t_atc = (time.perf_counter() - t0) / 20
        rows.append({"value": n, "exact_seconds": t_exact, "atc_seconds": t_atc})
    return rows


def setup_gap(n=8, seeds=C.SWEEP_SEEDS, n_families=C.DEFAULT_N_FAMILIES, setup_time=C.DEFAULT_SETUP_TIME):
    """Vehikel-B-Härtetest: ATC (mit dem ATCS-Rüstzeit-Term) gegen die echte Optimallösung MIT Rüstzeiten
    (CP-SAT, deshalb kleines n). Der Abstand ist eine echte Messfrage, kein behaupteter Befund."""
    gaps = []
    for seed in seeds:
        linst = SL.generate(n, seed, n_families=n_families, setup_time=setup_time)
        atc_order = A.atc_order(linst.p, linst.d, linst.w, linst.family, linst.setup)
        atc_total = A.evaluate_order(linst.p, linst.d, linst.w, atc_order, linst.family, linst.setup).total
        opt, proven = A.solve_exact(linst.p, linst.d, linst.w, linst.family, linst.setup, C.EXACT_TIME_LIMIT_SECONDS)
        if proven:
            gaps.append(_gap(atc_total, opt.total))
    return {"gap_mean": float(np.mean(gaps)), "gap_min": float(np.min(gaps)), "gap_max": float(np.max(gaps)), "n_runs": len(gaps)}


def setup_gap_sweep(setup_times=(0, 5, 15, 30, 60), n=8, seeds=C.SWEEP_SEEDS, n_families=C.DEFAULT_N_FAMILIES):
    return [{"value": s, **setup_gap(n=n, seeds=seeds, n_families=n_families, setup_time=s)} for s in setup_times]
