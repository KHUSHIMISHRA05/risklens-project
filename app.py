import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RiskLens | AI Risk Manager",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CONFIGURATION
# =========================================================

HF_REPO_ID = "khushii19/risklens-fraud-model"
HF_MODEL_FILENAME = "risk_model.pkl"

DATA_PATH = Path("data/risk_data.csv")

FEATURE_COLUMNS = [
    "Transaction Amount",
    "Payment Method",
    "Product Category",
    "Quantity",
    "Customer Age",
    "Device Used",
    "Account Age Days",
    "Transaction Hour",
]


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    /* =====================================================
       RISKLENS — COLORFUL FINTECH LIGHT UI
       ===================================================== */

    html, body, [data-testid="stAppViewContainer"] {
        background: #F4F7FC !important;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    section.main,
    section.main > div,
    [data-testid="stAppViewContainer"] .main {
        background: #F4F7FC !important;
        color: #18233F !important;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 5% 0%, rgba(115,87,255,.14), transparent 25%),
            radial-gradient(circle at 95% 0%, rgba(37,199,217,.12), transparent 25%),
            #F4F7FC !important;
    }

    .main .block-container {
        max-width: 1400px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #17213B 0%, #334D7B 58%, #2AA6B8 100%) !important;
        border: 0 !important;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #18233F !important;
    }

    .risklens-title {
        font-size: 3.2rem;
        line-height: 1;
        font-weight: 900;
        letter-spacing: -2px;
        background: linear-gradient(90deg, #253A68, #7357FF, #25AFC7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .risklens-subtitle {
        color: #71809D !important;
        font-size: 1.05rem;
        margin-bottom: 1rem;
    }

    .page-kicker {
        color: #7357FF !important;
        font-size: .75rem;
        font-weight: 900;
        letter-spacing: 1.5px;
    }

    .info-card, .decision-card {
        border: 1px solid #E1E7F0 !important;
        border-radius: 18px;
        padding: 1.15rem 1.25rem;
        background: #FFFFFF !important;
        color: #18233F !important;
        box-shadow: 0 10px 30px rgba(31,48,84,.07);
    }

    .info-card h4 {
        color: #71809D !important;
    }

    .info-card p {
        color: #18233F !important;
    }

    .hero-card {
        border-radius: 24px;
        padding: 1.7rem 2rem;
        background: linear-gradient(120deg, #253A68, #6852D9 52%, #25AFC7) !important;
        box-shadow: 0 18px 45px rgba(54,72,121,.18);
        margin: 1rem 0 1.3rem;
    }

    .hero-card h3, .hero-card p {
        color: #FFFFFF !important;
    }

    .risk-high {
        border-left: 5px solid #F45B78 !important;
        background: #FFF3F6 !important;
        padding: 1rem;
        border-radius: 14px;
    }

    .risk-medium {
        border-left: 5px solid #FF9D5C !important;
        background: #FFF8EF !important;
        padding: 1rem;
        border-radius: 14px;
    }

    .risk-low {
        border-left: 5px solid #35C98B !important;
        background: #F0FFF8 !important;
        padding: 1rem;
        border-radius: 14px;
    }

    /* =====================================================
       ALL STREAMLIT INPUTS — DARK TEXT, WHITE SURFACE
       ===================================================== */

    [data-testid="stNumberInput"],
    [data-testid="stSelectbox"],
    [data-testid="stTextInput"],
    [data-testid="stTextArea"],
    [data-testid="stSlider"] {
        color: #18233F !important;
    }

    [data-testid="stNumberInput"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stTextArea"] label,
    [data-testid="stSlider"] label,
    [data-testid="stWidgetLabel"] p {
        color: #354463 !important;
        -webkit-text-fill-color: #354463 !important;
        font-weight: 750 !important;
    }

    [data-testid="stNumberInput"] [data-baseweb="input"],
    [data-testid="stNumberInput"] [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input {
        background: #FFFFFF !important;
        color: #18233F !important;
        -webkit-text-fill-color: #18233F !important;
        opacity: 1 !important;
        color-scheme: light !important;
    }

    [data-testid="stNumberInput"] [data-baseweb="input"] {
        border: 1px solid #D7DFEC !important;
        border-radius: 13px !important;
    }

    [data-testid="stNumberInput"] button,
    [data-testid="stNumberInput"] button p,
    [data-testid="stNumberInput"] button svg {
        background: #FFFFFF !important;
        color: #354463 !important;
        fill: #354463 !important;
        -webkit-text-fill-color: #354463 !important;
        opacity: 1 !important;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"],
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stSelectbox"] [data-baseweb="select"] span,
    [data-testid="stSelectbox"] [data-baseweb="select"] p {
        background: #FFFFFF !important;
        color: #18233F !important;
        -webkit-text-fill-color: #18233F !important;
        opacity: 1 !important;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"] {
        border: 1px solid #D7DFEC !important;
        border-radius: 13px !important;
    }

    [data-testid="stSelectbox"] svg {
        fill: #354463 !important;
    }

    div[role="listbox"],
    ul[role="listbox"],
    div[role="option"] {
        background: #FFFFFF !important;
        color: #18233F !important;
        -webkit-text-fill-color: #18233F !important;
    }

    div[role="option"]:hover,
    div[role="option"][aria-selected="true"] {
        background: #F0EEFF !important;
        color: #5B43D6 !important;
    }

    [data-testid="stSlider"] [role="slider"] {
        background: #7357FF !important;
        border: 3px solid #FFFFFF !important;
    }

    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #7357FF, #4F8EFF, #25C7D9) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7357FF, #4F8EFF, #25C7D9) !important;
        color: #FFFFFF !important;
        border: 0 !important;
        border-radius: 13px !important;
        font-weight: 850 !important;
        box-shadow: 0 10px 24px rgba(88,91,220,.22) !important;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #E1E7F0 !important;
        border-radius: 18px !important;
        box-shadow: 0 9px 26px rgba(31,48,84,.06);
    }

    [data-testid="stMetricLabel"] {
        color: #71809D !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: #18233F !important;
        -webkit-text-fill-color: #18233F !important;
    }

    [data-testid="stAlert"],
    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        color: #18233F !important;
        border-color: #E1E7F0 !important;
        border-radius: 15px !important;
    }

    [data-testid="stDataFrame"] {
        border-radius: 14px !important;
        border: 1px solid #E1E7F0 !important;
    }

    hr {
        border-color: #E1E7F0 !important;
    }


    /* =====================================================
       PREMIUM RISK ASSESSMENT
       ===================================================== */
    .assessment-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin: 8px 0 18px;
    }

    .assessment-card {
        position: relative;
        overflow: hidden;
        background: #FFFFFF !important;
        border: 1px solid #E1E7F0 !important;
        border-radius: 20px;
        padding: 20px 22px;
        min-height: 118px;
        box-shadow: 0 10px 28px rgba(31,48,84,.07);
    }

    .assessment-card:after {
        content: "";
        position: absolute;
        width: 90px;
        height: 90px;
        border-radius: 50%;
        right: -32px;
        top: -35px;
        background: rgba(115,87,255,.08);
    }

    .assessment-label {
        color: #71809D !important;
        font-size: .78rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: .8px;
        margin-bottom: 8px;
    }

    .assessment-value {
        color: #18233F !important;
        font-size: 2rem;
        font-weight: 900;
        line-height: 1.05;
    }

    .assessment-value.high {
        color: #E74768 !important;
    }

    .assessment-value.medium {
        color: #E58A24 !important;
    }

    .assessment-value.low {
        color: #1BA975 !important;
    }

    .risk-meter {
        background: #FFFFFF !important;
        border: 1px solid #E1E7F0 !important;
        border-radius: 20px;
        padding: 18px 20px;
        margin: 6px 0 22px;
        box-shadow: 0 10px 28px rgba(31,48,84,.06);
    }

    .risk-meter-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #354463 !important;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .risk-meter-value {
        color: #7357FF !important;
        font-size: 1.05rem;
        font-weight: 900;
    }

    .risk-meter-track {
        height: 12px;
        background: #E8ECF4 !important;
        border-radius: 99px;
        overflow: hidden;
    }

    .risk-meter-fill {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #35C98B 0%, #FFB84D 50%, #F45B78 100%) !important;
    }

    .decision-premium {
        border-radius: 20px;
        padding: 22px 24px;
        margin: 6px 0 22px;
        background: linear-gradient(135deg, #FFF4F7, #FFF9FB) !important;
        border: 1px solid #FFD5DE !important;
        border-left: 6px solid #F45B78 !important;
        box-shadow: 0 10px 28px rgba(244,91,120,.08);
    }

    .decision-premium.medium {
        background: linear-gradient(135deg, #FFF8EF, #FFFCF7) !important;
        border-color: #FFE2B8 !important;
        border-left-color: #FF9D5C !important;
    }

    .decision-premium.low {
        background: linear-gradient(135deg, #F0FFF8, #F8FFFC) !important;
        border-color: #C8F1DF !important;
        border-left-color: #35C98B !important;
    }

    .decision-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(244,91,120,.12);
        color: #D83F61 !important;
        font-size: .74rem;
        font-weight: 900;
        letter-spacing: .7px;
    }

    .decision-premium.medium .decision-badge {
        background: rgba(255,157,92,.15);
        color: #C56F16 !important;
    }

    .decision-premium.low .decision-badge {
        background: rgba(53,201,139,.14);
        color: #168A5D !important;
    }

    .decision-premium h3 {
        color: #18233F !important;
        margin: 10px 0 6px;
        font-size: 1.45rem;
    }

    .decision-premium p {
        color: #53627D !important;
        margin: 0;
        font-size: 1rem;
    }

    .summary-wrap {
        background: #FFFFFF !important;
        border: 1px solid #E1E7F0 !important;
        border-radius: 20px;
        padding: 8px;
        box-shadow: 0 10px 28px rgba(31,48,84,.06);
    }

    @media (max-width: 900px) {
        .assessment-grid {
            grid-template-columns: 1fr;
        }
    }

    .risklens-footer {
        text-align: center;
        color: #8290A8 !important;
        font-size: .82rem;
        padding: 1.5rem 0 .25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource(show_spinner="Loading RiskLens AI model...")
def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
    )

    return joblib.load(model_path)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data(show_spinner=False)
def load_data():

    if not DATA_PATH.exists():
        return None

    try:

        df = pd.read_csv(DATA_PATH)

        if "Is Fraudulent" not in df.columns:
            return None

        if not all(
            column in df.columns
            for column in FEATURE_COLUMNS
        ):
            return None

        return df

    except Exception:
        return None


# =========================================================
# RISK FUNCTIONS
# =========================================================

def get_risk_level(probability):

    if probability >= 0.70:
        return "HIGH RISK"

    if probability >= 0.40:
        return "MEDIUM RISK"

    return "LOW RISK"


def get_recommendation(level):

    if level == "HIGH RISK":

        return (
            "Block or hold for manual review "
            "and verify the transaction."
        )

    if level == "MEDIUM RISK":

        return (
            "Perform additional verification "
            "before approval."
        )

    return (
        "Continue normal processing with "
        "standard fraud monitoring."
    )


# =========================================================
# CREATE TRANSACTION
# =========================================================

def make_transaction(
    amount,
    payment,
    product,
    quantity,
    age,
    device,
    account_age,
    hour,
):

    return pd.DataFrame(
        [
            {
                "Transaction Amount": amount,
                "Payment Method": payment,
                "Product Category": product,
                "Quantity": quantity,
                "Customer Age": age,
                "Device Used": device,
                "Account Age Days": account_age,
                "Transaction Hour": hour,
            }
        ]
    )[FEATURE_COLUMNS]


# =========================================================
# LOAD AI MODEL
# =========================================================

try:

    model = load_model()
    model_error = None

except Exception as exc:

    model = None
    model_error = exc


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🛡️ RiskLens")

    st.caption("AI Transaction Risk Manager")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🔍 Transaction Checker",
            "🧠 Explainability",
            "📈 Risk Analytics",
            "💰 Business Impact",
            "ℹ️ Model Information",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("#### Fraud Risk Intelligence")

    st.caption(
        "ML • SHAP • Cost-Aware Decisions"
    )

    if model is not None:

        st.success("AI Model Ready")

    else:

        st.error("AI Model Unavailable")

    st.divider()

    st.caption(
        "RiskLens • Razorpay AI Risk Manager"
    )


# =========================================================
# MODEL ERROR
# =========================================================

if model_error is not None:

    st.error(
        "RiskLens could not load the fraud model."
    )

    with st.expander("Technical details"):

        st.code(str(model_error))

    st.stop()


# =========================================================
# DASHBOARD
# =========================================================

if page == "📊 Dashboard":

    st.markdown(
        '<div class="risklens-title">'
        '🛡️ RiskLens'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="risklens-subtitle">'
        'AI-powered transaction fraud risk management'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    df = load_data()

    if df is not None:

        total = len(df)

        fraud = int(
            (df["Is Fraudulent"] == 1).sum()
        )

        genuine = int(
            (df["Is Fraudulent"] == 0).sum()
        )

        fraud_rate = (
            fraud / total * 100
            if total
            else 0
        )

        genuine_rate = (
            genuine / total * 100
            if total
            else 0
        )

        st.markdown(
            """
            <div class="hero-card">

            <h3>
            Real-time transaction risk intelligence
            </h3>

            <p>
            RiskLens combines machine learning,
            explainable AI and cost-aware decision
            support to identify potentially fraudulent
            transactions.
            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        cards = [

            (
                "Total Transactions",
                f"{total:,}",
            ),

            (
                "Fraud Transactions",
                f"{fraud:,}",
            ),

            (
                "Fraud Rate",
                f"{fraud_rate:.2f}%",
            ),

            (
                "Genuine Rate",
                f"{genuine_rate:.2f}%",
            ),
        ]

        cols = st.columns(4)

        for col, (label, value) in zip(
            cols,
            cards,
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="info-card">

                    <h4>{label}</h4>

                    <p>{value}</p>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            "### 🚨 Risk Intelligence"
        )

        left, right = st.columns(
            [1.45, 1]
        )

        with left:

            st.info(
                f"""
                **Fraud Exposure Snapshot**

                The dataset contains
                **{fraud:,}** fraudulent transactions
                out of **{total:,}** total transactions.

                **Fraud Rate:** {fraud_rate:.2f}%
                """
            )

        with right:

            st.markdown(
                "#### Transaction Health"
            )

            st.metric(
                "Genuine Transactions",
                f"{genuine:,}",
            )

            st.progress(
                genuine_rate / 100,
                text=f"{genuine_rate:.2f}% genuine",
            )

            st.metric(
                "Fraudulent Transactions",
                f"{fraud:,}",
            )

            st.progress(
                fraud_rate / 100,
                text=f"{fraud_rate:.2f}% fraudulent",
            )

        st.markdown(
            "### 📊 Transaction Distribution"
        )

        dist = pd.DataFrame(
            {
                "Transaction Type": [
                    "Genuine",
                    "Fraudulent",
                ],
                "Percentage": [
                    genuine_rate,
                    fraud_rate,
                ],
            }
        )

        fig, ax = plt.subplots(
            figsize=(9, 3.8)
        )

        ax.barh(
            dist["Transaction Type"],
            dist["Percentage"],
        )

        ax.set_xlabel(
            "Percentage of Transactions"
        )

        ax.set_xlim(0, 100)

        ax.set_title(
            "Genuine vs Fraudulent Transaction Distribution"
        )

        for i, value in enumerate(
            dist["Percentage"]
        ):

            ax.text(
                min(value + 1, 96),
                i,
                f"{value:.2f}%",
                va="center",
                fontweight="bold",
            )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.markdown(
            """
            <div class="hero-card">

            <h3>
            🛡️ RiskLens AI Risk Engine
            </h3>

            <p>
            The model is ready for individual
            transaction scoring. Dataset-level
            analytics require the evaluation dataset.
            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "Dataset-level analytics require "
            "`data/risk_data.csv`. Individual prediction "
            "remains available in Transaction Checker."
        )

        a, b, c = st.columns(3)

        with a:

            st.info(
                """
                **🤖 ML Risk Scoring**

                Fraud probability and
                0–100 risk score.
                """
            )

        with b:

            st.info(
                """
                **🧠 Explainable AI**

                SHAP-based feature attribution.
                """
            )

        with c:

            st.info(
                """
                **💰 Cost-Aware Decisions**

                Review and missed-fraud
                cost support.
                """
            )

    st.markdown(
        "### ⚡ Quick Actions"
    )

    a, b, c = st.columns(3)

    with a:

        st.markdown(
            """
            <div class="info-card">

            <h4>
            🔍 Transaction Checker
            </h4>

            <p style="font-size:1rem;font-weight:500">

            Analyze an individual transaction.

            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:

        st.markdown(
            """
            <div class="info-card">

            <h4>
            🧠 Explainability
            </h4>

            <p style="font-size:1rem;font-weight:500">

            Understand model risk drivers with SHAP.

            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with c:

        st.markdown(
            """
            <div class="info-card">

            <h4>
            💰 Business Impact
            </h4>

            <p style="font-size:1rem;font-weight:500">

            Translate predictions into
            cost-aware decisions.

            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "### 🧩 RiskLens Decision Pipeline"
    )

    st.markdown(
        """
        <div class="info-card">

        <p style="
        font-size:1.05rem;
        text-align:center;
        ">

        Transaction Data →
        ML Fraud Model →
        Risk Score →
        SHAP Explanation →
        Business Decision

        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# TRANSACTION CHECKER
# =========================================================

elif page == "🔍 Transaction Checker":

    st.markdown(
        "## 🔍 Transaction Risk Checker"
    )

    st.caption(
        "Estimate fraud probability and receive "
        "an actionable risk decision."
    )

    st.divider()

    left, right = st.columns(
        [1.3, 1]
    )

    with left:

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=2500.0,
            step=100.0,
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Credit Card",
                "Debit Card",
                "PayPal",
                "UPI",
                "Bank Transfer",
            ],
        )

        product = st.selectbox(
            "Product Category",
            [
                "Electronics",
                "Clothing",
                "Home",
                "Beauty",
                "Sports",
                "Other",
            ],
        )

        quantity = st.number_input(
            "Quantity",
            min_value=1,
            value=1,
            step=1,
        )

        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1,
        )

        device = st.selectbox(
            "Device Used",
            [
                "Desktop",
                "Mobile",
                "Tablet",
            ],
        )

        account_age = st.number_input(
            "Account Age Days",
            min_value=0,
            value=180,
            step=1,
        )

        hour = st.slider(
            "Transaction Hour",
            0,
            23,
            14,
        )

        analyze = st.button(
            "🔎 Analyze Transaction",
            type="primary",
            use_container_width=True,
        )

    with right:

        st.markdown(
            "### What RiskLens Provides"
        )

        st.markdown(
            """
            <div class="decision-card">

            <b>Fraud Probability</b>
            <br>
            Model-estimated probability.

            <br><br>

            <b>Risk Score</b>
            <br>
            0–100 score derived from probability.

            <br><br>

            <b>Risk Level</b>
            <br>
            HIGH, MEDIUM or LOW.

            <br><br>

            <b>Recommended Action</b>
            <br>
            Operational next step.

            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyze:

        tx = make_transaction(
            amount,
            payment,
            product,
            quantity,
            age,
            device,
            account_age,
            hour,
        )

        try:

            prob = float(
                model.predict_proba(tx)[0][1]
            )

            score = prob * 100

            level = get_risk_level(prob)

            st.divider()

            st.markdown(
                "### 🎯 Risk Assessment"
            )

            risk_class = {
                "HIGH RISK": "high",
                "MEDIUM RISK": "medium",
                "LOW RISK": "low",
            }[level]

            st.markdown(
                f"""
                <div class="assessment-grid">
                    <div class="assessment-card">
                        <div class="assessment-label">Fraud Probability</div>
                        <div class="assessment-value">{prob:.2%}</div>
                    </div>
                    <div class="assessment-card">
                        <div class="assessment-label">Risk Score</div>
                        <div class="assessment-value">{score:.1f} <span style="font-size:.95rem;color:#8290A8">/ 100</span></div>
                    </div>
                    <div class="assessment-card">
                        <div class="assessment-label">Risk Level</div>
                        <div class="assessment-value {risk_class}">{level}</div>
                    </div>
                </div>

                <div class="risk-meter">
                    <div class="risk-meter-top">
                        <span>Fraud risk intensity</span>
                        <span class="risk-meter-value">{prob:.2%}</span>
                    </div>
                    <div class="risk-meter-track">
                        <div class="risk-meter-fill" style="width:{score:.2f}%"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "### 🛡️ Decision"
            )

            decision_class = {
                "HIGH RISK": "",
                "MEDIUM RISK": "medium",
                "LOW RISK": "low",
            }[level]

            st.markdown(
                f"""
                <div class="decision-premium {decision_class}">
                    <span class="decision-badge">RECOMMENDED ACTION</span>
                    <h3>{level}</h3>
                    <p>{get_recommendation(level)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "### 📋 Transaction Summary"
            )

            summary = pd.DataFrame(
                {
                    "Field": FEATURE_COLUMNS,
                    "Value": [
                        amount,
                        payment,
                        product,
                        quantity,
                        age,
                        device,
                        account_age,
                        hour,
                    ],
                }
            )

            st.markdown(
                '<div class="summary-wrap">',
                unsafe_allow_html=True,
            )

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        except Exception as exc:

            st.error(
                "Unable to score this transaction."
            )

            with st.expander(
                "Technical details"
            ):

                st.code(str(exc))


# =========================================================
# EXPLAINABILITY
# =========================================================

elif page == "🧠 Explainability":

    st.markdown(
        "## 🧠 Explainable AI"
    )

    st.caption(
        "Understand which features influence "
        "an individual fraud-risk prediction."
    )

    st.divider()

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=2500.0,
        step=100.0,
        key="e_amount",
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Credit Card",
            "Debit Card",
            "PayPal",
            "UPI",
            "Bank Transfer",
        ],
        key="e_payment",
    )

    product = st.selectbox(
        "Product Category",
        [
            "Electronics",
            "Clothing",
            "Home",
            "Beauty",
            "Sports",
            "Other",
        ],
        key="e_product",
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=1,
        step=1,
        key="e_quantity",
    )

    age = st.number_input(
        "Customer Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
        key="e_age",
    )

    device = st.selectbox(
        "Device Used",
        [
            "Desktop",
            "Mobile",
            "Tablet",
        ],
        key="e_device",
    )

    account_age = st.number_input(
        "Account Age Days",
        min_value=0,
        value=180,
        step=1,
        key="e_account",
    )

    hour = st.slider(
        "Transaction Hour",
        0,
        23,
        14,
        key="e_hour",
    )

    explain = st.button(
        "🧠 Explain Risk",
        type="primary",
        use_container_width=True,
    )

    if explain:

        tx = make_transaction(
            amount,
            payment,
            product,
            quantity,
            age,
            device,
            account_age,
            hour,
        )

        try:

            # -------------------------------------------------
            # MODEL PREDICTION
            # -------------------------------------------------

            prob = float(
                model.predict_proba(tx)[0][1]
            )

            a, b, c = st.columns(3)

            a.metric(
                "Fraud Probability",
                f"{prob:.2%}",
            )

            b.metric(
                "Risk Score",
                f"{prob * 100:.1f} / 100",
            )

            c.metric(
                "Risk Level",
                get_risk_level(prob),
            )

            # -------------------------------------------------
            # GET PIPELINE COMPONENTS
            # -------------------------------------------------

            preprocessor = model.named_steps[
                "preprocessor"
            ]

            rf_model = model.named_steps[
                "model"
            ]

            # -------------------------------------------------
            # TRANSFORM INPUT
            # -------------------------------------------------

            transformed = preprocessor.transform(tx)

            if hasattr(
                transformed,
                "toarray",
            ):

                transformed = transformed.toarray()

            transformed = np.asarray(
                transformed
            )

            # -------------------------------------------------
            # SHAP
            # -------------------------------------------------

            explainer = shap.TreeExplainer(
                rf_model
            )

            shap_values = explainer.shap_values(
                transformed
            )

            # -------------------------------------------------
            # HANDLE SHAP OUTPUT
            # -------------------------------------------------

            if isinstance(
                shap_values,
                list,
            ):

                values = np.asarray(
                    shap_values[1]
                )

                values = values[0]

            else:

                shap_values = np.asarray(
                    shap_values
                )

                if shap_values.ndim == 3:

                    # samples, features, classes

                    values = shap_values[
                        0,
                        :,
                        1,
                    ]

                elif shap_values.ndim == 2:

                    if shap_values.shape[0] == 1:

                        values = shap_values[0]

                    elif shap_values.shape[1] == 2:

                        values = shap_values[:, 1]

                    else:

                        values = shap_values[0]

                else:

                    values = shap_values.ravel()

            values = np.asarray(
                values,
                dtype=float,
            ).ravel()

            # -------------------------------------------------
            # FEATURE NAMES
            # -------------------------------------------------

            try:

                names = list(
                    preprocessor.get_feature_names_out()
                )

            except Exception:

                names = [
                    f"Feature {i + 1}"
                    for i in range(len(values))
                ]

            if len(names) != len(values):

                names = [
                    f"Feature {i + 1}"
                    for i in range(len(values))
                ]

            # -------------------------------------------------
            # SHAP DATAFRAME
            # -------------------------------------------------

            exp = pd.DataFrame(
                {
                    "Feature": names,
                    "SHAP Value": values,
                }
            )

            exp["Absolute Impact"] = (
                exp["SHAP Value"].abs()
            )

            exp = exp.sort_values(
                "Absolute Impact",
                ascending=False,
            ).head(10)

            # -------------------------------------------------
            # TOP RISK DRIVERS
            # -------------------------------------------------

            st.markdown(
                "### 🔎 Top Risk Drivers"
            )

            st.dataframe(
                exp[
                    [
                        "Feature",
                        "SHAP Value",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            # -------------------------------------------------
            # SHAP CHART
            # -------------------------------------------------

            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            plot = exp.sort_values(
                "SHAP Value"
            )

            ax.barh(
                plot["Feature"],
                plot["SHAP Value"],
            )

            ax.axvline(
                0,
                linewidth=1,
            )

            ax.set_xlabel(
                "SHAP Value"
            )

            ax.set_title(
                "Top Features Influencing Fraud Prediction"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            st.caption(
                "Positive SHAP values push toward "
                "fraud; negative values push toward "
                "non-fraud."
            )

        except Exception as exc:

            st.error(
                "SHAP explanation could not be generated."
            )

            with st.expander(
                "Technical details"
            ):

                st.code(str(exc))


# =========================================================
# RISK ANALYTICS
# =========================================================

elif page == "📈 Risk Analytics":

    st.markdown(
        "## 📈 Risk Analytics"
    )

    st.caption(
        "Dataset-level fraud patterns and "
        "model risk distribution."
    )

    st.divider()

    df = load_data()

    if df is None:

        st.warning(
            "Risk Analytics requires "
            "`data/risk_data.csv`."
        )

    else:

        total = len(df)

        fraud = int(
            (df["Is Fraudulent"] == 1).sum()
        )

        genuine = int(
            (df["Is Fraudulent"] == 0).sum()
        )

        a, b, c = st.columns(3)

        a.metric(
            "Transactions",
            f"{total:,}",
        )

        b.metric(
            "Fraudulent",
            f"{fraud:,}",
        )

        c.metric(
            "Fraud Rate",
            f"{fraud / total * 100:.2f}%",
        )

        st.markdown(
            "### 📊 Dataset Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.bar(
            [
                "Genuine",
                "Fraudulent",
            ],
            [
                genuine,
                fraud,
            ],
        )

        ax.set_ylabel(
            "Transactions"
        )

        ax.set_title(
            "Genuine vs Fraudulent Transactions"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.markdown(
            "### 🧪 Model Risk Distribution"
        )

        try:

            sample = df.sample(
                n=min(20000, len(df)),
                random_state=42,
            )

            probs = model.predict_proba(
                sample[FEATURE_COLUMNS]
            )[:, 1]

            fig, ax = plt.subplots(
                figsize=(9, 4.5)
            )

            ax.hist(
                probs,
                bins=20,
            )

            ax.set_xlabel(
                "Fraud Probability"
            )

            ax.set_ylabel(
                "Transactions"
            )

            ax.set_title(
                "Fraud Probability Distribution "
                f"(sample of {len(sample):,})"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        except Exception as exc:

            st.error(
                "Unable to generate model "
                "risk distribution."
            )

            with st.expander(
                "Technical details"
            ):

                st.code(str(exc))


# =========================================================
# BUSINESS IMPACT
# =========================================================

elif page == "💰 Business Impact":

    st.markdown(
        "## 💰 Business Impact"
    )

    st.caption(
        "Translate model risk into "
        "cost-aware operational decisions."
    )

    st.divider()

    a, b, c = st.columns(3)

    with a:

        prob = st.slider(
            "Fraud Probability",
            0.0,
            1.0,
            0.83,
            0.01,
        )

    with b:

        review_cost = st.number_input(
            "Manual Review Cost (₹)",
            0.0,
            100000.0,
            100.0,
            10.0,
        )

    with c:

        missed_cost = st.number_input(
            "Missed Fraud Cost (₹)",
            0.0,
            100000.0,
            500.0,
            50.0,
        )

    score = prob * 100

    level = get_risk_level(
        prob
    )

    if level == "HIGH RISK":

        expected_review = review_cost

        action = "Manual review / hold"

    elif level == "MEDIUM RISK":

        expected_review = (
            review_cost * 0.5
        )

        action = "Additional verification"

    else:

        expected_review = 0

        action = "Normal processing"

    expected_missed = (
        prob * missed_cost
    )

    total_cost = (
        expected_review
        + expected_missed
    )

    a, b, c = st.columns(3)

    a.metric(
        "Risk Score",
        f"{score:.1f} / 100",
    )

    b.metric(
        "Risk Level",
        level,
    )

    c.metric(
        "Recommended Action",
        action,
    )

    st.markdown(
        "### 💸 Expected Cost Breakdown"
    )

    a, b, c = st.columns(3)

    a.metric(
        "Expected Review Cost",
        f"₹{expected_review:,.2f}",
    )

    b.metric(
        "Expected Missed-Fraud Cost",
        f"₹{expected_missed:,.2f}",
    )

    c.metric(
        "Estimated Total Cost",
        f"₹{total_cost:,.2f}",
    )

    cost_df = pd.DataFrame(
        {
            "Cost Type": [
                "Review Cost",
                "Expected Missed-Fraud Cost",
            ],
            "Amount": [
                expected_review,
                expected_missed,
            ],
        }
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        cost_df["Cost Type"],
        cost_df["Amount"],
    )

    ax.set_ylabel(
        "Estimated Cost (₹)"
    )

    ax.set_title(
        "Cost-Aware Risk Decision"
    )

    plt.xticks(
        rotation=10
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.info(
        "These monetary values are configurable "
        "business assumptions for demonstrating "
        "cost-aware decision support; they are not "
        "direct measured losses from the dataset."
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

else:

    st.markdown(
        "## ℹ️ Model Information"
    )

    st.caption(
        "Technical overview of the "
        "RiskLens fraud-risk engine."
    )

    st.divider()

    a, b, c = st.columns(3)

    a.metric(
        "Model Type",
        "Random Forest",
    )

    b.metric(
        "Explainability",
        "SHAP",
    )

    c.metric(
        "Output",
        "Risk Score 0–100",
    )

    st.markdown(
        "### 🔗 Model Source"
    )

    st.code(
        f"""
Hugging Face Repository:
{HF_REPO_ID}

Model File:
{HF_MODEL_FILENAME}
"""
    )

    st.markdown(
        "### 🧩 Input Features"
    )

    feature_info = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,

            "Type": [
                "Numeric",
                "Categorical",
                "Categorical",
                "Numeric",
                "Numeric",
                "Categorical",
                "Numeric",
                "Numeric",
            ],
        }
    )

    st.dataframe(
        feature_info,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### 🚦 Risk Thresholds"
    )

    thresholds = pd.DataFrame(
        {
            "Risk Level": [
                "LOW RISK",
                "MEDIUM RISK",
                "HIGH RISK",
            ],

            "Fraud Probability": [
                "< 40%",
                "40% – 69%",
                "≥ 70%",
            ],

            "Suggested Action": [
                "Normal processing",
                "Additional verification",
                "Manual review / hold",
            ],
        }
    )

    st.dataframe(
        thresholds,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### 🛡️ Project Positioning"
    )

    st.markdown(
        "RiskLens is an AI-powered transaction "
        "fraud risk-management prototype. "
        "It combines machine-learning fraud "
        "probability, 0–100 risk scoring, SHAP "
        "explainability, business impact estimation "
        "and cost-aware decision support."
    )
    