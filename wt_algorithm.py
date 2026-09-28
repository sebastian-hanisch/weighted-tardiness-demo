"""1||ΣwⱼTⱼ: n Aufträge mit Bearbeitungszeit pⱼ, Fälligkeit dⱼ und Gewicht wⱼ auf einer Maschine, Ziel ist die
GEWICHTETE Summe der Verspätungen zu minimieren (Tⱼ = max(0, Cⱼ - dⱼ)). Anders als SPT/EDD/Moore-Hodgson/WSPT
(Stücke 1-4 dieser Linie) gibt es HIER keine einfache Regel mehr, die beweisbar optimal ist - das Problem ist
stark NP-schwer (Lawler 1977; Lenstra, Rinnooy Kan & Brucker 1977), sobald Gewicht UND Frist gleichzeitig zählen.

Drei Bausteine:
1. **ATC** (Apparent Tardiness Cost, Vepsalainen & Morton 1987) - die literaturbekannte Dispatching-Heuristik,
   dieselbe Formel, die `warehouse-transfer-demo` bereits praktisch als ATCS (mit Rüstzeiten) einsetzt. Gut, aber
   NICHT bewiesen optimal.
2. **EDD/WSPT als falsche Regeln** - die beiden Vorgänger-Regeln dieser Linie, beide hier nicht mehr optimal
   (EDD ignoriert Gewichte, WSPT ignoriert Fälligkeiten).
3. **CP-SAT** (OR-Tools) als exakter Löser für die Gegenprobe - ein Kreis-Modell (wie `fahrzeugflotte-demo`s
   `fz_exact.py`), das Rüstzeiten beim Familienwechsel gleich mit abbildet: bei Rüstzeit 0 (neutrales Vehikel)
   kollabiert es exakt zum rüstzeitfreien Fall, kein zweites Modell nötig."""

import math
import os
from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model

NUM_SEARCH_WORKERS = min(8, os.cpu_count() or 1)  # mehr Worker als Kerne bremst CP-SAT eher (Oversubscription) -
# genau das schlug auf einem 4-Kern-CI-Runner mit einem hart auf 8 gesetzten Wert fehl (Timeout statt Beweis)

ATC_K = 0.9  # Look-ahead-Parameter (Vepsalainen & Morton 1987 empfehlen 0.5-4.5; 0.9 gegen das eigene
# Instanz-Schema (TF/RDD) über n=3..9 gegen CP-SAT gerastert, siehe README/Messreihe - dasselbe Vorgehen wie
# ATCS_K1/K2 in warehouse-transfer-demo)
ATC_K_SETUP = 0.5  # Look-ahead-Parameter für den Rüstzeit-Term (Vehikel B, ATCS-Erweiterung), ebenso gerastert


@dataclass
class Result:
    order: np.ndarray
    completion: np.ndarray
    tardiness: np.ndarray
    total: float          # ΣwⱼTⱼ


def _completion_times(p, order, family=None, setup=None):
    t = 0.0
    out = np.empty(len(order), dtype=np.float64)
    prev_family = None
    for idx, j in enumerate(order):
        if family is not None and prev_family is not None:
            t += float(setup[prev_family, family[j]])
        t += float(p[j])
        out[idx] = t
        prev_family = family[j] if family is not None else None
    return out


def evaluate_order(p, d, w, order, family=None, setup=None):
    order = np.asarray(order)
    completion = _completion_times(p, order, family, setup)
    tardiness = np.maximum(completion - np.asarray(d, dtype=np.float64)[order], 0.0)
    total = float(np.asarray(w, dtype=np.float64)[order] @ tardiness)
    return Result(order, completion, tardiness, total)


def edd_order(d):
    return np.argsort(d, kind="stable")


def wspt_order(p, w):
    ratio = np.asarray(w, dtype=np.float64) / np.asarray(p, dtype=np.float64)
    return np.argsort(-ratio, kind="stable")


def random_order(n, rng):
    order = np.arange(n)
    rng.shuffle(order)
    return order


# --- ATC (Apparent Tardiness Cost) -----------------------------------------------------------------------------


def atc_order(p, d, w, family=None, setup=None, k=ATC_K, k_setup=ATC_K_SETUP):
    """Dynamische Dispatching-Regel (Vepsalainen & Morton 1987): zu jedem Entscheidungszeitpunkt t den Auftrag
    mit dem höchsten Prioritätsindex I_j(t) = (w_j/p_j) * exp(-max(0, d_j - p_j - t) / (K * p̄)) wählen - p̄ ist
    die MITTLERE Bearbeitungszeit der noch nicht eingeplanten Aufträge, wird bei jedem Schritt neu berechnet.
    Anders als SPT/EDD/WSPT ist das KEINE statische Sortierung, sondern eine Simulation, die bei jedem Schritt
    neu entscheidet - der erste dynamische Algorithmus dieser Linie.

    Auf dem Werkstatt/Logistik-Vehikel (`family`/`setup` gesetzt) kommt ein zweiter Term dazu (ATCS-Erweiterung,
    dieselbe Formel wie in `warehouse-transfer-demo`): ein Familienwechsel wird zusätzlich mit
    exp(-Rüstzeit / (K_setup * p̄)) abgewertet, ein Auftrag der aktuellen Familie bevorzugt."""
    n = len(p)
    remaining = list(range(n))
    order = np.empty(n, dtype=np.int64)
    t = 0.0
    prev_family = None
    for step in range(n):
        p_bar = max(float(np.mean([p[j] for j in remaining])), 1e-9)
        best_j, best_score = None, -math.inf
        for j in remaining:
            slack = max(float(d[j]) - float(p[j]) - t, 0.0)
            score = (float(w[j]) / float(p[j])) * math.exp(-slack / (k * p_bar))
            if family is not None and prev_family is not None:
                s = float(setup[prev_family, family[j]])
                score *= math.exp(-s / (k_setup * p_bar))
            if score > best_score:
                best_j, best_score = j, score
        order[step] = best_j
        t += float(p[best_j])
        if family is not None and prev_family is not None:
            t += float(setup[prev_family, family[best_j]])
        prev_family = family[best_j] if family is not None else None
        remaining.remove(best_j)
    return order


def atc(p, d, w, family=None, setup=None):
    return evaluate_order(p, d, w, atc_order(p, d, w, family, setup), family, setup)


# --- CP-SAT (exakte Gegenprobe, ein Modell für beide Vehikel) ---------------------------------------------------


def solve_exact(p, d, w, family=None, setup=None, time_limit_seconds=10.0):
    """Ein-Maschinen-Sequenzierung mit ΣwⱼTⱼ-Ziel als Kreis-Modell: Knoten 0 ist ein Depot (Start UND Ende der
    Rundreise), Knoten 1..n sind die Aufträge. `AddCircuit` erzwingt eine Hamilton-Rundreise; die Rückkehr zum
    Depot ist kostenlos und trägt keine Zeitrestriktion. Zeitvariable je Auftrag wird NUR auf dem benutzten Bogen
    scharf gemacht (reified `>=`, kein `==` nötig - das Ziel ist in jeder Zeitvariable monoton steigend, der
    Löser drückt sie von selbst auf ihr Minimum). Bei Rüstzeit 0 (neutrales Vehikel) ist `setup` eine Nullmatrix -
    dasselbe Modell kollabiert dann exakt zum rüstzeitfreien Fall, kein zweites Modell nötig.

    Rückgabe: (Result oder None, proven: bool). `proven=False` heißt: Zeitlimit erreicht, das Ergebnis ist die
    beste gefundene (nicht notwendig optimale) Lösung; `None` nur wenn der Löser nicht einmal eine zulässige
    Lösung im Zeitlimit fand (bei diesem Problem praktisch nie, da immer zulässig)."""
    n = len(p)
    if family is None:
        family = np.zeros(n, dtype=np.int64)
        setup = np.zeros((1, 1), dtype=np.int64)
    horizon = int(np.sum(p)) + int(np.max(setup)) * n + 1

    m = cp_model.CpModel()
    time_vars = [m.NewIntVar(0, horizon, f"time_{j}") for j in range(n)]
    arcs = []
    for j in range(n):                                      # Depot -> Auftrag j: kein Rüstzeit-Term (erster Auftrag)
        lit = m.NewBoolVar(f"a0_{j}")
        arcs.append((0, j + 1, lit))
        m.Add(time_vars[j] >= p[j]).OnlyEnforceIf(lit)
        lit_back = m.NewBoolVar(f"a{j}_0")                   # Auftrag j -> Depot: Rückkehr, keine Zeitrestriktion
        arcs.append((j + 1, 0, lit_back))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lit = m.NewBoolVar(f"a{i}_{j}")
            arcs.append((i + 1, j + 1, lit))
            s = int(setup[family[i], family[j]])
            m.Add(time_vars[j] >= time_vars[i] + p[j] + s).OnlyEnforceIf(lit)
    m.AddCircuit(arcs)

    tard_terms = []
    for j in range(n):
        t_j = m.NewIntVar(0, horizon, f"tard_{j}")
        m.AddMaxEquality(t_j, [0, time_vars[j] - int(d[j])])
        tard_terms.append(int(w[j]) * t_j)
    m.Minimize(sum(tard_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = NUM_SEARCH_WORKERS
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, False

    times = np.array([solver.Value(v) for v in time_vars], dtype=np.float64)
    order = np.argsort(times, kind="stable")
    tardiness = np.maximum(times[order] - np.asarray(d, dtype=np.float64)[order], 0.0)
    total = float(np.asarray(w, dtype=np.float64)[order] @ tardiness)
    result = Result(order, times[order], tardiness, total)
    return result, status == cp_model.OPTIMAL
