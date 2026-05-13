"""
src/utils/visualiser.py

Generates Plotly figures for the Streamlit dashboard.
All functions return plotly Figure objects (not rendered — Streamlit calls
st.plotly_chart() on them).
"""

from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from config import TRAIT_NAMES, TRAITS, SCORE_MIN, SCORE_MAX


# ── Radar chart ───────────────────────────────────────────────────────────────

def radar_chart(
    scores: Dict[str, float],
    username: str,
    compare_scores: Optional[Dict[str, float]] = None,
    compare_label: str = "Comparison",
) -> go.Figure:
    """
    Big-Five radar chart.

    Parameters
    ----------
    scores         : {trait_abbr: score}
    username       : displayed in the chart title
    compare_scores : optional second user / population mean to overlay
    """
    labels = [TRAIT_NAMES[t] for t in TRAITS]
    values = [scores.get(t, 4.0) for t in TRAITS]
    # Close the polygon
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        name=f"u/{username}",
        line=dict(color="#636EFA", width=2),
        fillcolor="rgba(99, 110, 250, 0.25)",
    ))

    if compare_scores:
        cvals = [compare_scores.get(t, 4.0) for t in TRAITS]
        cvals_closed = cvals + [cvals[0]]
        fig.add_trace(go.Scatterpolar(
            r=cvals_closed,
            theta=labels_closed,
            fill="toself",
            name=compare_label,
            line=dict(color="#EF553B", width=2, dash="dot"),
            fillcolor="rgba(239, 85, 59, 0.15)",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[SCORE_MIN, SCORE_MAX],
                tickvals=[1, 2, 3, 4, 5, 6, 7],
                tickfont=dict(size=10),
            )
        ),
        showlegend=True,
        title=dict(text=f"Big-Five Profile — u/{username}", x=0.5),
        margin=dict(l=60, r=60, t=60, b=40),
        height=420,
    )
    return fig


# ── Bar chart ─────────────────────────────────────────────────────────────────

def bar_chart(scores: Dict[str, float], username: str) -> go.Figure:
    """Horizontal bar chart for at-a-glance score comparison."""
    trait_labels = [TRAIT_NAMES[t] for t in TRAITS]
    values       = [scores.get(t, 4.0) for t in TRAITS]
    colors       = ["#636EFA", "#00CC96", "#EF553B", "#AB63FA", "#FFA15A"]

    fig = go.Figure(go.Bar(
        x=values,
        y=trait_labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 7.5], title="Score (1–7)"),
        title=dict(text=f"Trait Scores — u/{username}", x=0.5),
        height=320,
        margin=dict(l=40, r=60, t=50, b=40),
        showlegend=False,
    )
    return fig


# ── Feature importance chart ──────────────────────────────────────────────────

def feature_importance_chart(
    feature_names: List[str],
    importances: List[float],
    trait: str,
    top_n: int = 15,
) -> go.Figure:
    """Top-N feature importances for a single trait's model."""
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.nlargest(top_n, "importance")

    fig = px.bar(
        df, x="importance", y="feature",
        orientation="h",
        title=f"Top {top_n} Features — {TRAIT_NAMES.get(trait, trait)}",
        color="importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ── Comment volume indicator ──────────────────────────────────────────────────

def reliability_gauge(comment_count: int) -> go.Figure:
    """Gauge showing estimated reliability based on comment volume."""
    # Rough heuristic: 200+ comments → full reliability
    reliability = min(comment_count / 200, 1.0) * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=reliability,
        number={"suffix": "%"},
        title={"text": f"Reliability estimate<br><sup>{comment_count} comments</sup>"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#636EFA"},
            "steps": [
                {"range": [0, 40],   "color": "#FFCDD2"},
                {"range": [40, 70],  "color": "#FFF9C4"},
                {"range": [70, 100], "color": "#C8E6C9"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 2},
                "thickness": 0.75,
                "value": 70,
            },
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
    return fig
