"""Vehikel B "Werkstatt/Logistik": dieselben Aufträge wie Vehikel A (Bearbeitungszeit, Fälligkeit, Gewicht),
zusätzlich eine Familie je Auftrag und eine feste Rüstzeit beim Familienwechsel (dieselbe Idee wie in
`spt-scheduling-demo`/`moore-hodgson-demo`/`wspt-demo`, dort bereits Vepsalainen & Morton 1987 zitiert - hier
zusätzlich load-bearing für den zweiten ATC-Term (ATCS-Erweiterung) UND für das CP-SAT-Kreis-Modell)."""

from dataclasses import dataclass

import numpy as np

import wt_constants as C


@dataclass(frozen=True)
class LogistikInstance:
    n: int
    p: np.ndarray
    d: np.ndarray
    w: np.ndarray
    family: np.ndarray
    setup: np.ndarray
    seed: int


def generate(n, seed, n_families=C.DEFAULT_N_FAMILIES, setup_time=C.DEFAULT_SETUP_TIME,
             tf=C.DEFAULT_TF, rdd=C.DEFAULT_RDD, p_min=C.P_MIN, p_max=C.P_MAX,
             weight_values=C.WEIGHT_VALUES, weight_probs=C.WEIGHT_PROBS):
    rng = np.random.default_rng(seed)
    p = rng.integers(p_min, p_max + 1, size=n).astype(np.int64)
    total = int(p.sum())
    frac = (1 - tf - rdd / 2) + rdd * rng.random(n)
    d = np.maximum(np.round(total * frac).astype(np.int64), p)
    w = rng.choice(weight_values, size=n, p=weight_probs).astype(np.int64)
    family = rng.integers(0, n_families, size=n).astype(np.int64)
    setup = np.full((n_families, n_families), setup_time, dtype=np.int64)
    np.fill_diagonal(setup, 0)
    return LogistikInstance(n, p, d, w, family, setup, seed)
