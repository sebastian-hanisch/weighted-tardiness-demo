# ATC – wenn keine einfache Regel mehr beweisbar optimal ist – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-weighted-tardiness-demo.streamlit.app/)**

Fünftes Stück der **Klassische-Scheduling-Theorie-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch
– Operations Research und Machine Learning": $n$ Aufträge mit Bearbeitungszeit $p_j$, Fälligkeit $d_j$ und Gewicht
$w_j$ auf **einer** Maschine, Ziel ist die GEWICHTETE Summe der Verspätungen zu minimieren
($1||\sum w_j T_j$, mit $T_j = \max(0, C_j - d_j)$).

**Einordnung in die Linie:** Das erste **stark NP-schwere** Stück (Lawler 1977; Lenstra, Rinnooy Kan & Brucker
1977) - anders als SPT, EDD, Moore-Hodgson und WSPT (Stücke 1-4) gibt es hier KEINE einfache Regel mehr, die
beweisbar optimal ist, sobald Frist UND Gewicht gleichzeitig zählen. Die beste bekannte Heuristik ist **ATC**
(Apparent Tardiness Cost, Vepsalainen & Morton 1987) - dieselbe Regel, die `warehouse-transfer-demo` bereits
praktisch als ATCS (mit Rüstzeiten) einsetzt. Hier zum ersten Mal ihr theoretisches Zuhause, mit CP-SAT
(OR-Tools) als exakter Gegenprobe für kleine Instanzen.
```
SPT (1||ΣCⱼ, Vertauschungsargument)                                              [Stück 1]
EDD (1||Lmax, dasselbe Beweismuster, andere Zielfunktion)                        [Stück 2]
Moore-Hodgson (1||ΣUⱼ, EDD + gezieltes Streichen)                                [Stück 3]
WSPT / Smith's Rule (1||ΣwⱼCⱼ, verallgemeinert SPT mit Gewichten)                [Stück 4]
 └─ ATC (1||ΣwⱼTⱼ, stark NP-schwer - erstes Stück ohne Beweis)                   [dieses Stück]
Johnson-Regel (F2||Cmax, zweite Maschine)                                        [Folgestück]
LPT (Pm||Cmax, parallele Maschinen)                                              [Folgestück]
Job Shop (Konvergenzpunkt: Reihenfolge UND Maschinenwahl)                        [Folgestück]
```

Ergebnis in Kürze: bei 20 Aufträgen mit Fristen UND Gewichten liegt ATC im Mittel **111,6 %** unter EDD (ignoriert
Gewichte), **189,6 %** unter WSPT (ignoriert Fristen) und **597,0 %** unter einer zufälligen Reihenfolge.
**Der ehrliche Bruch ist hier ein anderer als in Stück 1-4**: ATC trifft das CP-SAT-Optimum NICHT zuverlässig -
selbst OHNE Rüstzeiten (Trefferquote sinkt von 100 % bei n=2 auf 20 % bei n=9, mit erheblicher Streuung je
Instanz). Auf dem Werkstatt/Logistik-Vehikel wächst der Abstand zusätzlich mit der Rüstzeit (von **1,9 %** bei
0 Minuten auf **24,0 %** bei 60 Minuten je Familienwechsel) - und ATC kann dort sogar schlechter abschneiden als
die einfacheren, "falschen" Regeln EDD oder WSPT (im Einzelfall gemessen, siehe App).

| Frage | Ergebnis (Mittel über 5 feste Instanzen, Seeds 100000–100004, mit je 3 Ketten-Seeds) |
|---|---|
| Standardfall (20 Aufträge) | ✅ ATC liegt **111,6 %** unter EDD, **189,6 %** unter WSPT, **597,0 %** unter Zufall |
| **Trefferquote gegen CP-SAT (n=2..9)** | ❌ sinkt von 100 % auf 20 % - ATC ist eine Heuristik, KEIN Beweis |
| **Rechenzeit** | ➖ CP-SAT bei n=9 bereits mehrere Sekunden (Zeitlimit), ATC im Mikrosekundenbereich |
| **Vehikel Werkstatt/Logistik** | ❌ Rüstzeit 0/5/15/30/60 Minuten: **1,9/12,1/12,1/20,4/24,0 %** über dem Optimum |

## Was die Demo zeigt

1. **ATC in Aktion** (Schritt-Slider): **Aufträge** (Bearbeitungszeit, Farbe nach Gewicht) → **Einplanen**
   (Regler "eingeplante Aufträge", pünktlich/verspätet farblich markiert) → **Ergebnis** (Σ wⱼTⱼ-Kurve ATC gegen
   EDD).
2. **Was ATC bringt - und wo es aufhört:** ATC, EDD (falsche Regel hier), WSPT (falsche Regel hier), zufällige
   Reihenfolge, CP-SAT-Gegenprobe (n ≤ 9, mit Beweis-Status); auf dem Werkstatt/Logistik-Vehikel zusätzlich der
   ATCS-Term für Rüstzeiten.
3. **📐 Sweep** über die Anzahl der Aufträge.
4. **🔬 Experimente auf Abruf:** CP-SAT gegen ATC über n = 2 bis 9 (ehrliche Trefferquote, kein Beweis-Check);
   Rechenzeit CP-SAT (Zeitlimit) gegen ATC (O(n²)); Rüstzeit-Härtetest auf dem Werkstatt/Logistik-Vehikel.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" - die erste Zeile ist hier "es gibt eine
   einfache, beweisbar optimale Regel" selbst.

Regler: Aufträge (2–60), **Vehikel** (Neutral/Werkstatt-Logistik – bei Werkstatt zusätzlich Rüstzeit und Anzahl
Familien), Seed der Instanz (+ 🎲), Seed der Kette (+ 🎲, steuert nur die zufällige Vergleichsreihenfolge - ATC
selbst ist deterministisch).

## Die zwei Vehikel (gelten für die ganze Linie)

- **Neutral** (`wt_scenario.py`): $n$ Aufträge mit Bearbeitungszeit $p_j \sim U(1, 100)$, Fälligkeit $d_j$ nach
  dem TF/RDD-Schema (wie Stück 1-4) und Gewicht $w_j \in \{1,2,4\}$ (wie Stück 4) - hier zum ersten Mal BEIDES
  gleichzeitig load-bearing, genau das macht das Problem NP-schwer.
- **Werkstatt/Logistik** (`wt_scenario_logistik.py`): dieselben Aufträge, aber in Familien mit fester Rüstzeit
  beim Wechsel (wie in Stück 1/3/4). Rüstzeit 0 kollabiert strukturell exakt zum neutralen Vehikel (ATC liefert
  bitidentische Reihenfolgen, per Test belegt) - anders als in Stück 1-4 ist das aber KEIN Beweis, dass der
  Rüstzeit-0-Abstand zum Optimum automatisch null ist, weil ATC selbst schon ohne Rüstzeiten nicht bewiesen
  optimal ist (siehe "Was nicht funktioniert hat" unten).

## Modell und Verfahren

- **Instanz** (`wt_scenario.py`, `wt_scenario_logistik.py`): Bearbeitungszeiten, Fristen, Gewichte, Familien und
  Rüstzeit-Matrix, Seed-erzeugt wie jede andere Konzepte-Linie dieser Website.
- **ATC** (`wt_algorithm.py`): dynamische Dispatching-Regel (Vepsalainen & Morton 1987) - zu jedem
  Entscheidungszeitpunkt den Auftrag mit dem höchsten Prioritätsindex wählen, $K$ gegen die eigene
  Instanzverteilung gerastert (0,9 für den Fristen-Term, 0,5 für den Rüstzeit-Term - dasselbe Vorgehen wie
  ATCS_K1/K2 in `warehouse-transfer-demo`). Der erste dynamische (nicht statisch sortierende) Algorithmus dieser
  Linie.
- **CP-SAT** (`wt_algorithm.solve_exact`): ein Kreis-Modell (Depot-Knoten + ein Knoten je Auftrag, `AddCircuit`,
  wie `fahrzeugflotte-demo`s `fz_exact.py`) - EIN Modell für beide Vehikel, kollabiert bei Rüstzeit 0 automatisch
  zum rüstzeitfreien Fall.
- **Auswertung** (`wt_evaluation.py`): Kennzahlen, Sweep, Optimalitäts- und Timing-Messreihe, Rüstzeit-Härtetest.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutung: "ATC trifft das Optimum zuverlässig genug für eine 100-%-Trefferquote wie in Stück 1-4"**
  - **widerlegt, und zwar deutlich**: die Trefferquote gegen CP-SAT liegt zwischen 20 % und 100 % je nach n,
    ohne klaren Trend (n=9: 20 %, n=2: 100 %, dazwischen schwankend) - anders als bei den bewiesenen Regeln der
    Stücke 1-4 ist das hier der NORMALFALL, kein Fehler. Ein K-Parameter-Raster (0,5 bis 4,5) über das eigene
    Instanzschema hat die mittlere Abweichung von ~15 % auf ~4-5 % gedrückt, aber nicht auf null - genau das
    ist stark NP-schwer in der Praxis.
- **Der Rüstzeit-0-Fall ist KEIN Nulltest mehr**: in Stück 1-4 kollabierte die Vehikel-B-Härtekurve bei Rüstzeit
  0 automatisch auf 0 %, weil die Regel selbst bewiesen optimal war. Hier liegt der Rüstzeit-0-Abstand bereits
  bei ~1,9 % (derselbe heuristische Grundfehler wie ohne Vehikel B) - die KORREKTE Konsistenzprüfung ist
  stattdessen strukturell: ATCs Reihenfolge bei Rüstzeit 0 ist bitidentisch mit der auf dem neutralen Vehikel
  (per Test belegt), nicht "trifft das Optimum".
- **CP-SAT ist nur bis $n \approx 9$ praktikabel** - deutlich früher als die Brute-Force-Grenze der Stücke 1-4
  (auch $n=9$), aber aus einem anderen Grund: nicht $n!$ Permutationen, sondern die stark NP-schwere Struktur
  selbst (gemessene Zeitlimit-Kante bei n=10-11, siehe Timing-Experiment).
- **Synthetische Instanzen:** Bearbeitungszeiten gleichverteilt, Fristen nach dem TF/RDD-Schema, Gewichte aus
  einer festen Verteilung, keine Präzedenzen, ein Auftrag = eine Operation.

## Verifikation

- **CP-SAT gegen unabhängige Brute-Force-Vollaufzählung** (mit und ohne Rüstzeiten): für n = 3 bis 7 über
  mehrere Seeds stimmt das Kreis-Modell exakt mit einer unabhängigen `itertools.permutations`-Vollaufzählung
  überein - der exakte Löser selbst ist geprüft, nicht nur behauptet.
- **Struktureller Konsistenz-Test**: ATCs Reihenfolge auf dem Werkstatt/Logistik-Vehikel mit Rüstzeit 0 ist
  bitidentisch mit der auf dem neutralen Vehikel.
- **Handrechnung:** eine kleine, von Hand nachgerechnete Instanz (2 Aufträge, ein wichtiger mit knapper Frist
  gegen einen unwichtigen mit lockerer Frist) bestätigt, dass ATC den wichtigen zuerst einplant.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Standardfall, Trefferquote gegen CP-SAT, Rechenzeit,
  Rüstzeit-Härtetest; positive **und** negative Aussagen inklusive der Tatsache, dass ATC NICHT zuverlässig
  optimal ist); alle 5 Presets geprüft; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt auf
  beiden Vehikeln, Würfel-Knöpfe, Permalink-Grenzen inkl. ungültigem Vehikel, Extremwerte, Experimente auf
  Abruf, Footer, korrekt formatierte negative Prozent-Abstände); eigener Test für die ausblendbaren Regler
  (kein verwaister Widget-Zustand nach Permalink/Preset - von Anfang an eingebaut, siehe
  [[feedback_hideable_slider_desync_recurs_in_new_demos]]).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweep, 🔬 Experimente, 🚧 Grenzen, Mathe |
| `wt_algorithm.py` | ATC, EDD/WSPT als falsche Regeln, CP-SAT (Kreis-Modell, ein Modell für beide Vehikel) |
| `wt_scenario.py` | Vehikel Neutral |
| `wt_scenario_logistik.py` | Vehikel Werkstatt/Logistik (Familien, Rüstzeit-Matrix) |
| `wt_constants.py` | Konstanten, Presets |
| `wt_evaluation.py` | Kennzahlen, Sweep, Optimalitäts- und Timing-Messreihe, Rüstzeit-Härtetest |
| `wt_presets.py`, `wt_visualization.py` | Permalink/Presets (inkl. `seed_widget`/`KEPT` für ausblendbare Regler), Plotly-Figuren (achsengesperrt) |
| `tests/` | CP-SAT gegen Vollaufzählung, Szenario und Auswertung, Aussagen der App, Presets, versteckter Widget-Zustand, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
