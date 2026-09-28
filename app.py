"""ATC (Apparent Tardiness Cost) - wenn keine einfache Regel mehr beweisbar optimal ist - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der neuen Konzepte-Linie "Klassische Scheduling-Theorie": n Aufträge auf einer Maschine, jeder mit
Fälligkeit UND Gewicht, Ziel ist die GEWICHTETE Summe der Verspätungen zu minimieren (1||Σwⱼ Tⱼ in der
α|β|γ-Notation). Anders als SPT/EDD/Moore-Hodgson/WSPT (Stücke 1-4) gibt es HIER keine beweisbar optimale
Sortierregel mehr - das Problem ist stark NP-schwer (Lawler 1977; Lenstra, Rinnooy Kan & Brucker 1977), sobald
Frist UND Gewicht gleichzeitig zählen. Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import wt_constants as C
from wt_evaluation import Settings, SWEEP_LABELS, analyse, instance, optimality_check, run_config, setup_gap, setup_gap_sweep, sweep, timing_sweep
from wt_presets import KEPT, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_chain_seed, randomize_seed, seed_widget, sync_query_params
from wt_visualization import build_jobs_chart, build_schedule, build_setup_gap, build_sweep, build_tardiness_curve, build_timing

st.set_page_config(page_title="ATC – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _optimality():
    return optimality_check()


@st.cache_data(show_spinner=False)
def _timing():
    return timing_sweep()


@st.cache_data(show_spinner=False)
def _setup_gap_sweep(n, n_families):
    return setup_gap_sweep(n=n, n_families=n_families)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


def _fmt_pct(x):
    """Vorzeichen-korrekt: `+95.3 %` (Vergleichsregel schlechter als ATC) oder `-4.2 %` (ATC selbst ist nur
    eine Heuristik, KEIN Beweis - eine Vergleichsregel kann hier auch mal zufällig besser abschneiden, anders
    als in Stück 1-4 dieser Linie)."""
    return f"{x:+.1f} %"


st.title("🎯 ATC – wenn keine einfache Regel mehr beweisbar optimal ist")
st.markdown(
    r"""
**n Aufträge auf einer Maschine, jeder mit Fälligkeit UND Gewicht, gesucht ist die Reihenfolge, die die
GEWICHTETE Summe der Verspätungen minimiert** ($1||\sum w_j T_j$, mit $T_j = \max(0, C_j - d_j)$). Anders als bei
SPT, EDD, Moore-Hodgson und WSPT (Stücke 1-4 dieser Linie) gibt es hier **keine einfache Regel mehr, die beweisbar
optimal ist** - das Problem ist stark NP-schwer (Lawler 1977), sobald Frist und Gewicht gleichzeitig zählen. Die
beste bekannte einfache Regel ist **ATC** (Apparent Tardiness Cost, Vepsalainen & Morton 1987): zu jedem
Zeitpunkt den Auftrag mit dem höchsten Prioritätsindex wählen - eine gute Heuristik, aber **kein Beweis**.
"""
)
st.caption(
    "Fünftes Stück der Konzepte-Linie „Klassische Scheduling-Theorie“ - das erste stark NP-schwere. Zwei "
    "Vehikel: **Neutral** (Aufträge mit Bearbeitungszeit, Frist und Gewicht) und **Werkstatt/Logistik** (dieselben "
    "Aufträge, aber in Familien mit Rüstzeit beim Wechsel) - der Umschalter ist in der Seitenleiste."
)

with st.expander("So funktioniert ATC", expanded=True):
    st.markdown(
        r"""
1. **Zu jedem Zeitpunkt neu entscheiden.** Anders als SPT/EDD/WSPT ist ATC KEINE statische Sortierung - bei jedem freien Maschinenplatz wird unter den noch offenen Aufträgen neu gewählt: $I_j(t) = (w_j/p_j) \cdot \exp(-\max(0, d_j - p_j - t) / (K \bar p))$, mit $\bar p$ = mittlere Bearbeitungszeit der noch offenen Aufträge.
2. **Warum kein Beweis.** Das Problem ist stark NP-schwer - es gibt (wenn P≠NP) kein Sortierkriterium, das in polynomieller Zeit immer optimal ist. ATC ist eine der besten bekannten Heuristiken, aber eine Heuristik.
3. **Was gemessen wird.** Der Abstand einer Reihenfolge zu ATC in Prozent - **kann hier auch negativ werden**, wenn eine einfachere Regel zufällig besser abschneidet. Für kleine $n$ zusätzlich CP-SAT (OR-Tools) als exakte Gegenprobe.
4. **Die Grenze der Annahme.** ATC selbst kennt keine Rüstzeiten. Das Vehikel „Werkstatt/Logistik“ prüft, was passiert, wenn Familienwechsel zusätzlich Zeit kosten.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
cols = st.columns(len(preset_names))
for col, name in zip(cols, preset_names):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_jobs = st.slider("Aufträge", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
                        help=f"Anzahl der Aufträge. Bis {C.EXACT_MAX_N} löst CP-SAT das Problem live mit (kann einige Sekunden dauern - stark NP-schwer).")
    vehicle = st.radio("Vehikel", list(C.VEHICLE_LABELS), key="vehicle_radio", format_func=lambda k: C.VEHICLE_LABELS[k],
                        help="Neutral: nur Bearbeitungszeiten, Fristen, Gewichte. Werkstatt/Logistik: dieselben Aufträge, zusätzlich in Familien mit Rüstzeit beim Wechsel.")
    if vehicle == "logistik":
        seed_widget("setup_time_slider")
        setup_time = st.slider("Rüstzeit je Familienwechsel (Minuten)", *bounds("setup_time_slider"), key="setup_time_slider",
                                help="0 Minuten kollabiert exakt zum neutralen Vehikel (siehe Test/Messreihe).")
        st.session_state[KEPT["setup_time_slider"]] = setup_time
        seed_widget("n_families_slider")
        n_families = st.slider("Auftragsfamilien", *bounds("n_families_slider"), key="n_families_slider",
                                help="Weniger Familien bei gleicher Auftragszahl bedeutet mehr Wechsel und damit mehr Rüstzeit insgesamt.")
        st.session_state[KEPT["n_families_slider"]] = n_families
    else:
        setup_time = int(st.session_state.get(KEPT["setup_time_slider"], C.DEFAULT_SETUP_TIME))
        n_families = int(st.session_state.get(KEPT["n_families_slider"], C.DEFAULT_N_FAMILIES))
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für Bearbeitungszeiten, Fristen und Gewichte.")
    chain_seed = st.number_input("Zufalls-Seed der Kette", *bounds("chain_seed_input"), key="chain_seed_input", step=1,
                                  help="Steuert nur die zufällige Vergleichs-Reihenfolge - ATC selbst ist deterministisch (kein Zufall im Kern).")
    st.button("🎲 Neue Kette würfeln", width="stretch", on_click=randomize_chain_seed, help="Würfelt einen neuen Seed für die Zufalls-Vergleichsreihenfolge.")

sync_query_params({"n_slider": int(n_jobs), "seed_input": int(seed), "chain_seed_input": int(chain_seed), "vehicle_radio": vehicle,
                    "setup_time_slider": int(setup_time), "n_families_slider": int(n_families)})

settings = Settings(int(n_jobs), int(seed), int(chain_seed), vehicle=vehicle, setup_time=int(setup_time), n_families=int(n_families))
with st.spinner("Rechne... (bei kleinem n löst CP-SAT live mit, das kann einige Sekunden dauern)"):
    a = _analysis(settings)
inst = a.inst
p, d, w = inst.p, inst.d, inst.w
data_key = settings

# --- ATC in Aktion ---------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 ATC in Aktion")
STEP_LABELS = {1: "1 · Aufträge", 2: "2 · Einplanen", 3: "3 · Ergebnis"}
if "wt_step" not in st.session_state or st.session_state.get("wt_step_owner") != data_key:
    st.session_state["wt_step"] = 1
    st.session_state["wt_step_owner"] = data_key
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="wt_step", format_func=lambda s: STEP_LABELS[s])

if step == 2:
    it_col, itplay_col = st.columns([5, 2])
    with it_col:
        upto = st.slider("Eingeplante Aufträge", 1, int(n_jobs), value=int(n_jobs), key="wt_upto")
else:
    upto = int(n_jobs)

view_slot = st.empty()
with view_slot.container():
    if step == 1:
        st.markdown(f"**{n_jobs} Aufträge, unsortiert** (Bearbeitungszeit in Minuten, Farbe nach Gewicht)")
        st.plotly_chart(build_jobs_chart(p, w), width="stretch", key="s1_jobs")
    elif step == 2:
        st.markdown(f"**ATC-Reihenfolge nach {upto} von {n_jobs} Aufträgen** (rot = verspätet)")
        st.plotly_chart(build_schedule(p, a.atc.order, a.atc.completion, a.atc.tardiness, upto=upto), width="stretch", key=f"s2_sched_{upto}")
        st.caption(f"Σ wⱼTⱼ bisher: {_fmt_int((w[a.atc.order][:upto] * a.atc.tardiness[:upto]).sum())}")
    else:
        st.markdown("**ATC gegen EDD (ignoriert Gewichte): Σ wⱼTⱼ über die Zeit**")
        st.plotly_chart(build_tardiness_curve(w, a.atc.order, a.atc.tardiness, a.edd.order, a.edd.tardiness), width="stretch", key="s3_curve")

if step == 1:
    st.caption(f"Bearbeitungszeiten zwischen {int(p.min())} und {int(p.max())} Minuten, Gewichte {sorted(set(w.tolist()))} (Seed {seed}).")
elif step == 2:
    gap_note = " Lücken zwischen Balken sind Rüstzeit bei einem Familienwechsel." if vehicle == "logistik" else ""
    st.caption(f"Jeder Balken ist ein Auftrag; rot markiert einen verspäteten Auftrag (Cⱼ > dⱼ).{gap_note}")
else:
    st.caption(f"ATC: Σ wⱼTⱼ {_fmt_int(a.atc.total)}. EDD (ignoriert Gewichte): {_fmt_int(a.edd.total)} (Differenz {_fmt_pct(a.gap_edd)}).")

st.markdown("---")

# --- Ergebnis -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was ATC bringt - und wo es aufhört")
vehicle_note = " Auf dem Werkstatt/Logistik-Vehikel zählt die Rüstzeit beim Familienwechsel mit - ATC bekommt dafür einen zweiten Term." if vehicle == "logistik" else ""
st.caption(f"**Abstand:** Σ wⱼTⱼ einer Reihenfolge gegenüber ATC in Prozent - kann negativ werden, ATC ist keine bewiesen optimale Regel.{vehicle_note}")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("ATC (Σ wⱼTⱼ)", _fmt_int(a.atc.total), help="Die Zielgröße: gewichtete Summe aller Verspätungen in ATC-Reihenfolge, auf dem gewählten Vehikel.")
m2.metric("EDD (ignoriert Gewichte)", _fmt_pct(a.gap_edd), delta_color="off", help="Die Regel aus Stück 2 dieser Linie - hier falsch, weil sie Gewichte ignoriert.")
m3.metric("WSPT (ignoriert Fristen)", _fmt_pct(a.gap_wspt), delta_color="off", help="Die Regel aus Stück 4 dieser Linie - hier falsch, weil sie Fristen ignoriert.")
m4.metric(f"Zufällige Reihenfolge (Mittel über {a.random_runs})", _fmt_pct(a.gap_random), delta_color="off", help="Mittel über mehrere zufällige Reihenfolgen derselben Instanz.")
if a.optimal is not None and a.optimal_proven:
    m5.metric("CP-SAT (exakte Gegenprobe)", "trifft ATC exakt" if a.atc_matches_optimum else "ATC weicht ab", delta_color="off",
              help="OR-Tools CP-SAT hat das Problem für diese Instanz bewiesen exakt gelöst (auf dem gewählten Vehikel).")
elif a.optimal is not None:
    m5.metric("CP-SAT", "Zeitlimit erreicht", delta_color="off", help="CP-SAT hat innerhalb des Zeitlimits keine bewiesen optimale Lösung gefunden - das gezeigte Ergebnis ist nur die beste GEFUNDENE, kein Beweis.")
else:
    m5.metric("CP-SAT", f"erst ab n ≤ {C.EXACT_MAX_N}", delta_color="off", help="Bei dieser Größe wäre eine exakte Lösung aussichtslos - das Problem ist stark NP-schwer (siehe Experiment unten).")

if a.optimal is not None and a.optimal_proven and not a.atc_matches_optimum:
    gap = 100.0 * (a.atc.total - a.optimal.total) / max(a.optimal.total, 1e-9) if a.optimal.total > 1e-9 else float(a.atc.total - a.optimal.total)
    st.warning(f"⚠️ ATC trifft hier NICHT das Optimum: {gap:.1f} % darüber (CP-SAT hat das bewiesen). Das ist normal - ATC ist eine Heuristik, kein Beweis.")
elif vehicle == "logistik" and (a.gap_edd < 0 or a.gap_wspt < 0 or a.gap_random < 0):
    worst = min(a.gap_edd, a.gap_wspt, a.gap_random)
    worse_than = "EDD" if a.gap_edd == worst else ("WSPT" if a.gap_wspt == worst else "eine zufällige Reihenfolge")
    st.warning(f"⚠️ Auf diesem Werkstatt-Vehikel schneidet ATC hier sogar schlechter ab als {worse_than}: {abs(worst):.1f} % mehr. Kein Fehler - ATC ist eine Heuristik, keine bewiesen optimale Regel.")
else:
    tail = " (CP-SAT bestätigt: das ist hier sogar das exakte Optimum, aber nicht garantiert)" if a.optimal is not None and a.optimal_proven and a.atc_matches_optimum else ""
    st.success(f"✅ ATC ist {a.gap_edd:.1f} % besser als EDD, {a.gap_wspt:.1f} % besser als WSPT und {a.gap_random:.1f} % besser als eine zufällige Reihenfolge{tail}.")

st.markdown("---")

# --- Sweeps -----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt der Vorsprung von der Instanzgröße ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
if st.button("Sweep über 5 feste Instanzen berechnen (dauert wenige Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {sweep_param}
if sweep_param in st.session_state.get("sweep_done", set()):
    rows_sweep = _sweep(sweep_param, Settings())
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")
    st.caption("Mittel über 5 feste Instanzen (Seeds 100000–100004) mit je drei Zufalls-Ketten für die Vergleichsreihenfolge.")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie oft trifft ATC das echte Optimum?")
if st.button(f"CP-SAT über n = 2 bis {C.EXACT_MAX_N} berechnen (dauert typisch 20-30 Sekunden)", key="opt_start"):
    st.session_state["opt_on"] = True
if st.session_state.get("opt_on"):
    rows_opt = _optimality()
    st.table({"Aufträge": [r["value"] for r in rows_opt], "Trefferquote": [f"{r['match_rate']:.0%}" for r in rows_opt], "davon bewiesen": [f"{r['proven_rate']:.0%}" for r in rows_opt]})
    st.caption("Für jede Instanzgröße 5 feste Instanzen: ATC gegen CP-SAT (nur über bewiesene Läufe gemittelt). Anders als bei SPT/EDD/Moore-Hodgson/WSPT ist hier KEIN 100-%-Ergebnis zu erwarten - ATC ist eine Heuristik, kein Beweis.")

st.markdown("---")

st.subheader("🔬 Wie teuer ist eine exakte Lösung wirklich?")
if st.button(f"Rechenzeit für n = 2 bis {C.EXACT_MAX_N} messen (dauert typisch 10-15 Sekunden)", key="timing_start"):
    st.session_state["timing_on"] = True
if st.session_state.get("timing_on"):
    rows_t = _timing()
    st.plotly_chart(build_timing(rows_t), width="stretch", key="timing_chart")
    last = rows_t[-1]
    st.caption(f"Bei {last['value']} Aufträgen braucht CP-SAT bereits {last['exact_seconds']*1000:.0f} ms, ATC {last['atc_seconds']*1000:.3f} ms. Anders als bei n! (Stück 1-4): hier wächst die Rechenzeit im schlimmsten Fall exponentiell, weil das Problem stark NP-schwer ist - kein pseudopolynomieller Trick hilft.")

st.markdown("---")

st.subheader("🔬 Werkstatt/Logistik: bleibt ATC gut, wenn Rüstzeiten dazukommen?")
if st.button("Rüstzeit von 0 bis 60 Minuten durchfahren (dauert typisch 15-25 Sekunden)", key="setup_start"):
    st.session_state["setup_on"] = True
if st.session_state.get("setup_on"):
    rows_s = _setup_gap_sweep(min(int(n_jobs), C.EXACT_MAX_N), int(n_families))
    st.plotly_chart(build_setup_gap(rows_s), width="stretch", key="setup_chart")
    st.caption("ATC bekommt zwar einen zweiten Term für Rüstzeiten (ATCS-Erweiterung), verglichen mit der echten Optimallösung MIT Rüstzeiten (CP-SAT, deshalb kleine Instanz). Anders als bei Stück 1-4: der Abstand ist bei Rüstzeit 0 NICHT automatisch null - ATC ist schon ohne Rüstzeiten keine bewiesen optimale Regel.")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Es gibt eine einfache, beweisbar optimale Regel** | Sobald Frist UND Gewicht gleichzeitig zählen, gibt es (wenn P≠NP) keine solche Regel mehr - ATC ist die beste bekannte Heuristik, kein Beweis. | Kein direkter Nachfolger in dieser Linie - der Punkt dieses Stücks |
| **Die Bearbeitungszeit hängt nicht von der Reihenfolge ab** | Sobald Rüstzeiten zwischen Auftragsfamilien dazukommen (Vehikel „Werkstatt/Logistik“), wächst der Abstand zum echten Optimum weiter (siehe Experiment oben). | Kein direkter Nachfolger in dieser Linie |
| **Es gibt nur eine Maschine** | Mit mehreren Maschinen wird aus einer Sortierfrage eine Zuordnungs- UND Reihenfolgefrage. | **Johnson-Regel (2 Maschinen), LPT (parallele Maschinen), Job Shop** (Folgestücke) |
"""
)
st.caption(
    "Fünftes Stück der Linie „Klassische Scheduling-Theorie“: das erste stark NP-schwere - verallgemeinert "
    "WSPT (Stück 4) um Fristen. Verwandt: `warehouse-transfer-demo` (ATCS-Regel praktisch, mit Rüstzeiten), "
    "die Exakte-Suche-Linie (CP-SAT als Standard-Werkzeug für NP-schwere Probleme)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem** ($1||\sum w_j T_j$): $n$ Aufträge mit Bearbeitungszeit $p_j$, Fälligkeit $d_j$ und Gewicht $w_j$ auf
einer Maschine; eine Reihenfolge $\pi$ legt die Fertigstellungszeit $C_j$ und damit die Verspätung
$T_j = \max(0, C_j - d_j)$ jedes Auftrags fest. Gesucht: $\pi$, das $\sum_j w_j T_j$ minimiert.

**Komplexität.** Stark NP-schwer (Lawler 1977; Lenstra, Rinnooy Kan & Brucker 1977) - anders als bei $\sum C_j$
(SPT), $L_{\max}$ (EDD) oder $\sum U_j$ (Moore-Hodgson) gibt es KEIN Vertauschungsargument, das eine einfache
Sortierregel als global optimal beweist: die Tardiness-Funktion $\max(0, \cdot)$ ist nicht linear, benachbarte
Vertauschungen können sich nicht mehr sauber gegeneinander aufrechnen.

**ATC** (Vepsalainen & Morton 1987): zu jedem Entscheidungszeitpunkt $t$ den Auftrag mit dem höchsten Index
$I_j(t) = (w_j/p_j) \cdot \exp(-\max(0, d_j - p_j - t)/(K\bar p))$ wählen, $\bar p$ = mittlere Bearbeitungszeit
der noch offenen Aufträge, neu berechnet bei jedem Schritt. $K$ ist gegen die eigene Instanzverteilung gerastert
(siehe README) - eine Simulation, keine statische Sortierung.

**CP-SAT-Modell** (`solve_exact`): ein Kreis-Modell über Depot-Knoten 0 und Auftrags-Knoten 1..n (`AddCircuit`),
Zeitvariable je Auftrag wird nur auf dem benutzten Bogen scharf gemacht; bei Rüstzeit 0 kollabiert es exakt zum
rüstzeitfreien Fall - ein Modell für beide Vehikel.

**Kennzahl.** Abstand zu ATC $= 100 \cdot (\text{Wert} - \text{Wert}_{\text{ATC}}) / \text{Wert}_{\text{ATC}}$ -
anders als in Stück 1-4 KANN das negativ werden. Für $n \le 9$ zusätzlich CP-SAT als exakte Gegenprobe (mit
Zeitlimit, da stark NP-schwer).

Implementiert in `wt_algorithm.py` (ATC, EDD/WSPT als falsche Regeln, CP-SAT), `wt_scenario.py`/
`wt_scenario_logistik.py` (die zwei Vehikel), `wt_evaluation.py` (Kennzahlen, Sweep, Optimalitäts- und
Timing-Messreihe, Rüstzeit-Härtetest).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
