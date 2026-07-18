import streamlit as st


def apply_global_styles() -> None:
    """Apply a modern analytics UI theme that keeps Streamlit text visible."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #111827;
            --muted: #64748b;
            --soft: #f8fafc;
            --panel: #ffffff;
            --line: #e2e8f0;
            --line-strong: #cbd5e1;
            --brand: #2563eb;
            --brand-dark: #1d4ed8;
            --teal: #0f766e;
            --green: #15803d;
            --red: #b91c1c;
            --amber: #b45309;
            --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        }

        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1280px;
            padding: 2.25rem 2.25rem 3.5rem;
        }

        h1, h2, h3, h4, h5, h6,
        p, li, label, span, div, .stMarkdown {
            color: var(--ink);
        }

        h1 {
            color: #0f172a;
            font-size: clamp(2rem, 4vw, 3.15rem);
            font-weight: 850;
            letter-spacing: 0;
            line-height: 1.04;
            margin-bottom: 0.45rem;
        }

        h2, h3 {
            color: #0f172a;
            font-weight: 800;
            letter-spacing: 0;
        }

        h2 {
            margin-top: 1.4rem;
        }

        h3 {
            font-size: 1.2rem;
            margin-top: 1.1rem;
        }

        .app-subtitle {
            color: var(--muted);
            font-size: 1rem;
            font-weight: 500;
            line-height: 1.55;
            margin-bottom: 1.75rem;
            max-width: 760px;
        }

        section[data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        section[data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }

        section[data-testid="stSidebar"] h2 {
            color: #ffffff !important;
            font-size: 1.15rem;
            letter-spacing: 0;
            margin-bottom: 0.2rem;
        }

        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: #94a3b8 !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 8px;
            margin: 0.18rem 0;
            padding: 0.38rem 0.55rem;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(37, 99, 235, 0.16);
            border-color: rgba(96, 165, 250, 0.32);
        }

        section[data-testid="stSidebar"] div[data-testid="stAlert"] {
            background: rgba(15, 118, 110, 0.12);
            border: 1px solid rgba(45, 212, 191, 0.26);
        }

        .soft-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: var(--shadow);
            min-height: 148px;
            padding: 1.15rem 1.2rem;
        }

        .soft-card h4 {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0;
            margin: 0 0 0.65rem;
            text-transform: uppercase;
        }

        .soft-card .card-value {
            color: #0f172a;
            font-size: 1.45rem;
            font-weight: 850;
            line-height: 1.2;
            margin-bottom: 0.5rem;
        }

        .soft-card p {
            color: var(--muted);
            font-size: 0.92rem;
            font-weight: 450;
            line-height: 1.45;
            margin: 0;
        }

        .section-note {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-left: 4px solid var(--brand);
            border-radius: 8px;
            color: var(--ink);
            font-weight: 520;
            line-height: 1.55;
            padding: 0.95rem 1.05rem;
        }

        .flow-box {
            align-items: center;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: var(--shadow);
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            padding: 1rem;
        }

        .flow-step {
            background: #f8fafc;
            border: 1px solid var(--line);
            border-radius: 8px;
            color: #0f172a;
            flex: 1 1 150px;
            font-size: 0.9rem;
            font-weight: 700;
            padding: 0.8rem 0.9rem;
            text-align: center;
        }

        .result-panel {
            border-radius: 8px;
            color: #ffffff;
            margin-bottom: 1rem;
            padding: 1.35rem;
            box-shadow: var(--shadow);
        }

        .result-panel * {
            color: #ffffff !important;
        }

        .result-panel.green {
            background: linear-gradient(135deg, #047857, #0f766e);
        }

        .result-panel.red {
            background: linear-gradient(135deg, #991b1b, #dc2626);
        }

        .risk-badge {
            border-radius: 8px;
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0;
            padding: 0.32rem 0.58rem;
            text-transform: uppercase;
        }

        .risk-low {
            background: #dcfce7;
            color: var(--green) !important;
        }

        .risk-medium {
            background: #fef3c7;
            color: var(--amber) !important;
        }

        .risk-high {
            background: #fee2e2;
            color: var(--red) !important;
        }

        div[data-testid="stForm"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: var(--shadow);
            padding: 1.15rem;
        }

        div[data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        button[data-baseweb="tab"] {
            background: #f8fafc;
            border: 1px solid var(--line);
            border-radius: 8px;
            color: #334155;
            font-weight: 700;
            padding: 0.35rem 0.8rem;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: var(--brand);
            border-color: var(--brand);
        }

        button[data-baseweb="tab"][aria-selected="true"] p {
            color: #ffffff !important;
        }

        .stButton > button,
        div[data-testid="stFormSubmitButton"] button {
            background: var(--brand);
            border: 0;
            border-radius: 8px;
            color: #ffffff;
            font-weight: 800;
            min-height: 2.9rem;
        }

        .stButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            background: var(--brand-dark);
            color: #ffffff;
        }

        input, textarea,
        div[data-baseweb="select"] *,
        div[data-baseweb="input"] * {
            color: var(--ink) !important;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"],
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            color: var(--ink);
            padding: 0.35rem;
        }

        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        textarea {
            background: #ffffff !important;
            border-color: var(--line-strong) !important;
            border-radius: 8px !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
            color: var(--ink);
        }

        div[data-testid="stAlert"] * {
            color: var(--ink) !important;
        }

        .js-plotly-plot,
        div[data-testid="stPlotlyChart"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: var(--shadow);
            padding: 0.45rem;
        }

        hr {
            border-color: rgba(148, 163, 184, 0.25);
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 1.35rem;
            }

            .soft-card {
                min-height: 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
