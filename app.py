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

    .main {
        padding-top: 1rem;
    }

    .risklens-title {
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .risklens-subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }

    .info-card {
        border: 1px solid rgba(100,116,139,.20);
        border-radius: 14px;
        padding: 1.15rem;
        background: rgba(255,255,255,.82);
        min-height: 115px;
        box-shadow: 0 2px 10px rgba(15,23,42,.04);
    }

    .info-card h4 {
        margin: 0 0 .45rem;
        color: #475569;
        font-size: .92rem;
    }

    .info-card p {
        margin: 0;
        color: #0f172a;
        font-size: 1.55rem;
        font-weight: 800;
    }

    .hero-card {
        border-radius: 18px;
        padding: 1.35rem 1.5rem;
        background: linear-gradient(
            135deg,
            #eff6ff 0%,
            #f8fafc 100%
        );
        border: 1px solid #dbeafe;
        margin-bottom: 1rem;
    }

    .hero-card h3 {
        margin: 0;
        font-size: 1.3rem;
    }

    .hero-card p {
        margin: .4rem 0 0;
        color: #475569;
    }

    .risk-high {
        border-left: 6px solid #dc2626;
        background: #fef2f2;
        padding: 1rem;
        border-radius: 10px;
    }

    .risk-medium {
        border-left: 6px solid #d97706;
        background: #fffbeb;
        padding: 1rem;
        border-radius: 10px;
    }

    .risk-low {
        border-left: 6px solid #16a34a;
        background: #f0fdf4;
        padding: 1rem;
        border-radius: 10px;
    }

    .decision-card {
        border: 1px solid rgba(100,116,139,.20);
        border-radius: 14px;
        padding: 1.15rem;
        background: #fff;
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

            a, b, c = st.columns(3)

            a.metric(
                "Fraud Probability",
                f"{prob:.2%}",
            )

            b.metric(
                "Risk Score",
                f"{score:.1f} / 100",
            )

            c.metric(
                "Risk Level",
                level,
            )

            st.progress(
                prob,
                text=f"Fraud probability: {prob:.2%}",
            )

            st.markdown(
                "### 🛡️ Decision"
            )

            css = {
                "HIGH RISK": "risk-high",
                "MEDIUM RISK": "risk-medium",
                "LOW RISK": "risk-low",
            }[level]

            st.markdown(
                f"""
                <div class="{css}">

                <h3>{level}</h3>

                <p>
                {get_recommendation(level)}
                </p>

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

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
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
    