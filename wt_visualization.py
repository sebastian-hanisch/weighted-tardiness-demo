"""Plotly-Abbildungen der Gewichtete-Verspätung-Demo: Auftragsübersicht (Bearbeitungszeit, Gewicht),
Gantt-artiges Balkendiagramm (pünktlich/verspätet eingefärbt), gewichtete Verspätungs-Kurve, Sweep, Timing.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

WEIGHT_COLORS = {1: "#9ecae9", 2: "#4c78a8", 4: "#e45756"}
ON_TIME_COLOR = "#4c78a8"
LATE_COLOR = "#e45756"
EDD_COLOR = "#54a24b"
WSPT_COLOR = "#f2cf5b"
RANDOM_COLOR = "#7f7f7f"
SETUP_COLOR = "#f58518"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_jobs_chart(p, w):
    fig = go.Figure()
    for wt in sorted(set(w.tolist())):
        idx = np.where(w == wt)[0]
        fig.add_trace(go.Bar(x=idx.tolist(), y=p[idx].tolist(), marker_color=WEIGHT_COLORS.get(wt, "#4c78a8"),
                              name=f"Gewicht {wt}", hovertemplate="Auftrag %{x}<br>Dauer %{y}<extra></extra>"))
    fig.update_xaxes(title_text="Auftrag (unsortiert)")
    fig.update_yaxes(title_text="Bearbeitungszeit")
    return _base(fig, 260)


def build_schedule(p, order, completion, tardiness, upto=None):
    """Balken je Auftrag in der gegebenen Reihenfolge, eingefärbt nach pünktlich/verspätet (Tⱼ = 0 bzw. > 0) -
    zeigt direkt, was die Zielfunktion zählt. `completion` sind die TATSÄCHLICHEN Fertigstellungszeiten - auf
    dem Werkstatt/Logistik-Vehikel enthalten sie Lücken durch Rüstzeiten."""
    order = np.asarray(order)
    upto = len(order) if upto is None else upto
    starts = np.asarray(completion) - p[order]
    fig = go.Figure()
    shown_on_time = shown_late = False
    for i in range(upto):
        j = order[i]
        is_late = bool(tardiness[i] > 1e-9)
        color = LATE_COLOR if is_late else ON_TIME_COLOR
        name = "verspätet" if is_late else "pünktlich"
        showlegend = not (shown_late if is_late else shown_on_time)
        if is_late:
            shown_late = True
        else:
            shown_on_time = True
        fig.add_trace(go.Bar(x=[float(p[j])], y=["Maschine"], base=[float(starts[i])], orientation="h",
                              marker=dict(color=color, line=dict(width=1, color="white")),
                              name=name, showlegend=showlegend, hovertemplate=f"Auftrag {j}<br>Dauer {p[j]}<extra></extra>"))
    fig.update_xaxes(title_text="Zeit")
    fig.update_yaxes(showticklabels=False)
    return _base(fig, 180)


def build_tardiness_curve(w, atc_order, atc_tardiness, edd_order, edd_tardiness):
    """Kumulierte gewichtete Verspätung Σ wⱼTⱼ über die Position - ATC gegen EDD (ignoriert Gewichte)."""
    x = np.arange(1, len(atc_tardiness) + 1)
    atc_cum = np.cumsum(w[atc_order] * atc_tardiness)
    edd_cum = np.cumsum(w[edd_order] * edd_tardiness)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=atc_cum, mode="lines+markers", line=dict(color=ON_TIME_COLOR, width=2.5), name="ATC (kumulierte gewichtete Verspätung)"))
    fig.add_trace(go.Scatter(x=x, y=edd_cum, mode="lines+markers", line=dict(color=EDD_COLOR, width=2, dash="dash"), name="EDD (ignoriert Gewichte)"))
    fig.update_xaxes(title_text="Aufträge eingeplant")
    fig.update_yaxes(title_text="Σ wⱼTⱼ bisher")
    return _base(fig, 320)


def build_sweep(rows, param_label, value_key="value",
                 y_keys=(("gap_edd", "ATC gegen EDD (ignoriert Gewichte)", EDD_COLOR), ("gap_wspt", "ATC gegen WSPT (ignoriert Fristen)", WSPT_COLOR), ("gap_random", "ATC gegen Zufall", RANDOM_COLOR))):
    xs = [r[value_key] for r in rows]
    fig = go.Figure()
    for key, name, color in y_keys:
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="lines+markers", line=dict(color=color, width=2.5), name=name))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Abstand zu ATC (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 360)


def build_timing(rows):
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["exact_seconds"] * 1000 for r in rows], mode="lines+markers", line=dict(color=LATE_COLOR, width=2.5), name="CP-SAT (Zeitlimit, im schlimmsten Fall exponentiell)"))
    fig.add_trace(go.Scatter(x=xs, y=[r["atc_seconds"] * 1000 for r in rows], mode="lines+markers", line=dict(color=ON_TIME_COLOR, width=2.5), name="ATC (O(n²))"))
    fig.update_xaxes(title_text="Aufträge")
    fig.update_yaxes(title_text="Rechenzeit (ms)", type="log")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 340)


def build_setup_gap(rows):
    xs = [r["value"] for r in rows]
    upper = [r["gap_max"] for r in rows]
    lower = [r["gap_min"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=upper + lower[::-1], fill="toself", fillcolor="rgba(245,133,24,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["gap_mean"] for r in rows], mode="lines+markers", line=dict(color=SETUP_COLOR, width=2.5), name="ATC über dem echten Optimum (CP-SAT)"))
    fig.update_xaxes(title_text="Rüstzeit je Familienwechsel (Minuten)")
    fig.update_yaxes(title_text="Abstand zum Optimum (%)")
    return _base(fig, 340)
