"""Konstanten der Gewichtete-Verspätung-Demo: beide Vehikel (Neutral, Werkstatt/Logistik), Regler,
Messreihen-Seeds."""

N_MIN, N_MAX, DEFAULT_N, N_STEP = 2, 60, 20, 1
SEED_MAX = 999999
DEFAULT_SEED = 45
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_CHAINS = 3

# Bearbeitungszeiten
P_MIN, P_MAX = 1, 100

# Fälligkeiten (TF/RDD-Schema, wie Stück 1-4, hier load-bearing)
DEFAULT_TF, DEFAULT_RDD = 0.4, 0.6

# Gewichte (wie Stück 4, hier ZUSAMMEN mit Fristen load-bearing - das macht das Problem NP-schwer)
WEIGHT_VALUES = (1, 2, 4)
WEIGHT_PROBS = (0.5, 0.3, 0.2)

# CP-SAT exakte Gegenprobe: praktisches Limit für eine live nutzbare Demo (gemessen, siehe README) - anders als
# bei Stück 1-4 nicht durch n! (Brute-Force), sondern durch die starke NP-Schwere selbst gesetzt.
EXACT_MAX_N = 9
EXACT_SWEEP_N = (2, 3, 4, 5, 6, 7, 8, 9)
EXACT_TIME_LIMIT_SECONDS = 10.0

# --- Vehikel B "Werkstatt/Logistik" ---------------------------------------------------------------------------
N_FAMILIES_MIN, N_FAMILIES_MAX, DEFAULT_N_FAMILIES = 2, 6, 3
SETUP_TIME_MIN, SETUP_TIME_MAX, DEFAULT_SETUP_TIME = 0, 60, 15

VEHICLE_LABELS = {"neutral": "Neutral", "logistik": "Werkstatt/Logistik"}
DEFAULT_VEHICLE = "neutral"


def _preset(n=DEFAULT_N, vehicle=DEFAULT_VEHICLE, setup_time=DEFAULT_SETUP_TIME, n_families=DEFAULT_N_FAMILIES):
    return {"n": n, "seed": DEFAULT_SEED, "chain_seed": 0, "vehicle": vehicle, "setup_time": setup_time, "n_families": n_families}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Kleine Instanz (CP-SAT sichtbar)": _preset(n=EXACT_MAX_N),
    "Große Instanz (Skalierung)": _preset(n=N_MAX),
    "Werkstatt/Logistik-Vehikel": _preset(vehicle="logistik"),
    "Hohe Rüstlast (Werkstatt)": _preset(vehicle="logistik", setup_time=SETUP_TIME_MAX),
}
# Werte in PRESET_HELP nach der Messreihe (wt_evaluation.run_config) final eingetragen.
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "20 Aufträge mit Fristen UND Gewichten: ATC misst sich gegen EDD (ignoriert Gewichte), WSPT (ignoriert Fristen) und Zufall.",
    "Kleine Instanz (CP-SAT sichtbar)": f"{EXACT_MAX_N} Aufträge: hier löst CP-SAT (OR-Tools) das Problem exakt mit - dauert je nach Instanz bis zu {EXACT_TIME_LIMIT_SECONDS:.0f} Sekunden, weil das Problem stark NP-schwer ist.",
    "Große Instanz (Skalierung)": f"{N_MAX} Aufträge: ATC bleibt schnell (O(n²)), eine exakte Lösung wäre bei dieser Größe aussichtslos.",
    "Werkstatt/Logistik-Vehikel": "Dieselben Aufträge, aber in Familien mit Rüstzeit beim Wechsel - ATC bekommt dafür einen zweiten Term (ATCS-Erweiterung).",
    "Hohe Rüstlast (Werkstatt)": f"Rüstzeit {SETUP_TIME_MAX} Minuten je Familienwechsel: der Abstand zwischen ATC und den falschen Regeln wächst.",
}
