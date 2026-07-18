import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(__file__), ".matplotlib_cache"),
)

import streamlit as st

from frontend.pages import (
    render_about_page,
    render_dashboard_page,
    render_fairness_metrics_page,
    render_model_comparison_page,
    render_predict_applicant_page,
    render_shap_explanation_page,
)
from frontend.styles import apply_global_styles


APP_TITLE = "Fair Credit Intelligence"
APP_SUBTITLE = "Explainable credit risk scoring with TabNet predictions, SHAP evidence, and fairness monitoring."


def render_sidebar() -> str:
    """Render the sidebar and return the selected navigation item."""
    with st.sidebar:
        st.markdown("## FairScore")
        st.caption("Risk, fairness, and explanations in one workspace.")

        st.markdown("---")
        selected_page = st.radio(
            "Workspace",
            [
                "Dashboard",
                "Predict Applicant",
                "SHAP Explanation",
                "Fairness Metrics",
                "Model Comparison",
                "About Project",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.info(
            "Research demonstration only. Do not use this app for live lending decisions."
        )

    return selected_page


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="bar_chart",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_global_styles()

    selected_page = render_sidebar()

    st.markdown(f"# {APP_TITLE}")
    st.markdown(f"<p class='app-subtitle'>{APP_SUBTITLE}</p>", unsafe_allow_html=True)

    if selected_page == "Dashboard":
        render_dashboard_page()
    elif selected_page == "Predict Applicant":
        render_predict_applicant_page()
    elif selected_page == "SHAP Explanation":
        render_shap_explanation_page()
    elif selected_page == "Fairness Metrics":
        render_fairness_metrics_page()
    elif selected_page == "Model Comparison":
        render_model_comparison_page()
    elif selected_page == "About Project":
        render_about_page()


if __name__ == "__main__":
    main()
