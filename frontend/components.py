from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_card(title: str, value: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="soft-card">
            <h4>{title}</h4>
            <div class="card-value">{value}</div>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, status: str, explanation: str) -> None:
    badge_class = {
        "Fair": "risk-low",
        "Needs Review": "risk-medium",
        "High Bias Risk": "risk-high",
    }.get(status, "risk-medium")

    st.markdown(
        f"""
        <div class="soft-card">
            <h4>{title}</h4>
            <div class="card-value">{value}</div>
            <span class="risk-badge {badge_class}">{status}</span>
            <p style="margin-top: 0.75rem;">{explanation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_panel(prediction: dict) -> None:
    decision = prediction["prediction"]
    confidence = float(prediction["confidence"])
    creditworthy_probability = float(prediction.get("creditworthy_probability", confidence))
    decision_threshold = prediction.get("decision_threshold")
    decision_margin = prediction.get("decision_margin")
    risk_level = prediction["risk_level"]
    interpretation = prediction["interpretation"]

    panel_class = "green" if decision == "Creditworthy" else "red"
    risk_class = {
        "Low Risk": "risk-low",
        "Medium Risk": "risk-medium",
        "High Risk": "risk-high",
    }.get(risk_level, "risk-medium")

    st.markdown(
        f"""
        <div class="result-panel {panel_class}">
            <div style="font-size: 0.78rem; opacity: 0.86; text-transform: uppercase; font-weight: 800;">
                Final Prediction
            </div>
            <div style="font-size: 2.2rem; font-weight: 850; line-height: 1.1; margin-top: 0.32rem;">{decision}</div>
            <div style="font-size: 0.98rem; line-height: 1.55; margin-top: 0.7rem;">{interpretation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([2, 1])
    with left:
        st.write(f"Creditworthy probability: **{creditworthy_probability:.1%}**")
        st.progress(creditworthy_probability)
        if decision_threshold is not None:
            margin_text = (
                f"Decision margin: **{float(decision_margin):.1%}**"
                if decision_margin is not None
                else ""
            )
            st.caption(
                f"Decision threshold: {float(decision_threshold):.1%}. {margin_text}"
            )
    with right:
        st.markdown(
            f"<span class='risk-badge {risk_class}'>{risk_level}</span>",
            unsafe_allow_html=True,
        )


def render_flow(steps: Iterable[str]) -> None:
    step_markup = "".join(f"<div class='flow-step'>{step}</div>" for step in steps)
    st.markdown(
        f"<div class='flow-box'>{step_markup}</div>",
        unsafe_allow_html=True,
    )


def render_shap_bar_chart(shap_values: pd.DataFrame) -> None:
    sorted_values = shap_values.sort_values("contribution", ascending=True)
    colors = ["#dc2626" if value < 0 else "#0f766e" for value in sorted_values["contribution"]]

    fig = go.Figure(
        go.Bar(
            x=sorted_values["contribution"],
            y=sorted_values["feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(
        height=420,
        font=dict(color="#111827", family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=10, r=10, t=24, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Feature contribution",
        yaxis_title=None,
    )
    fig.update_xaxes(gridcolor="#e2e8f0", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")


def render_grouped_metric_chart(comparison_data: pd.DataFrame) -> None:
    long_data = comparison_data.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Value",
    )
    fig = px.bar(
        long_data,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        color_discrete_sequence=["#64748b", "#2563eb", "#0f766e"],
    )
    fig.update_layout(
        height=430,
        font=dict(color="#111827", family="Inter, Segoe UI, sans-serif"),
        legend_title_text="",
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Score / Difference",
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="#e2e8f0", zeroline=False)
    st.plotly_chart(fig, width="stretch")
