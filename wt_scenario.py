"""Vehikel A "Neutral" der Gewichtete-Verspätung-Demo: n Aufträge mit Bearbeitungszeit pⱼ, Fälligkeit dⱼ (nach
dem literaturüblichen TF/RDD-Schema, wie Stück 1-4) und Gewicht wⱼ ∈ {1,2,4} (wie Stück 4, WSPT) auf EINER
Maschine. Anders als WSPT (Stück 4) ist hier BEIDES zugleich load-bearing - Frist UND Gewicht -, genau das macht
das Problem NP-schwer (Lawler 1977; Lenstra, Rinnooy Kan & Brucker 1977)."""

from dataclasses import dataclass

import numpy as np

import wt_constants as C


@dataclass(frozen=True)
class Instance:
    n: int
    p: np.ndarray
    d: np.ndarray
    w: np.ndarray
    seed: int
    tf: float
    rdd: float


def generate(n, seed, tf=C.DEFAULT_TF, rdd=C.DEFAULT_RDD, p_min=C.P_MIN, p_max=C.P_MAX,
             weight_values=C.WEIGHT_VALUES, weight_probs=C.WEIGHT_PROBS):
    rng = np.random.default_rng(seed)
    p = rng.integers(p_min, p_max + 1, size=n).astype(np.int64)
    total = int(p.sum())
    frac = (1 - tf - rdd / 2) + rdd * rng.random(n)
    d = np.maximum(np.round(total * frac).astype(np.int64), p)
    w = rng.choice(weight_values, size=n, p=weight_probs).astype(np.int64)
    return Instance(n, p, d, w, seed, tf, rdd)
