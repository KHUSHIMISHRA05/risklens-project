import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


# -------------------------------------------------
# Load trained model
# -------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/risk_model.pkl")


model = load_model()


# -------------------------------------------------
# Page configuration
# -------------------------------------------------

st.set_page_config(
    page_title="RiskLens",
    page_icon="🛡️",
    layout="wide"
)


# -------------------------------------------------
# Title
# -------------------------------------------------

st.title("🛡️ RiskLens")
st.subheader("AI-Powered Transaction Fraud Risk Manager")

st.write(
    "Enter transaction details below to estimate fraud risk "
    "and understand why the transaction was classified as risky."
)

st.divider()


# -------------------------------------------------
# Transaction Input
# -------------------------------------------------

st.header("Transaction Details")

col1, col2 = st.columns(2)


with col1:

    transaction_amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=850.50
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "credit card",
            "debit card",
            "PayPal",
            "bank transfer"
        ]
    )

    product_category = st.selectbox(
        "Product Category",
        [
            "electronics",
            "clothing",
            "home & garden",
            "toys & games",
            "health & beauty",
            "sports & fitness"
        ]
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=2,
        step=1
    )


with col2:

    customer_age = st.number_input(
        "Customer Age",
        min_value=1,
        max_value=100,
        value=30
    )

    device_used = st.selectbox(
        "Device Used",
        [
            "mobile",
            "desktop",
            "tablet"
        ]
    )

    account_age_days = st.number_input(
        "Account Age (Days)",
        min_value=0,
        value=15
    )

    transaction_hour = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=2
    )


st.divider()


# -------------------------------------------------
# Analyze Transaction
# -------------------------------------------------

if st.button("🔍 Analyze Transaction", width="stretch"):

    transaction = {

        "Transaction Amount": transaction_amount,
        "Payment Method": payment_method,
        "Product Category": product_category,
        "Quantity": quantity,
        "Customer Age": customer_age,
        "Device Used": device_used,
        "Account Age Days": account_age_days,
        "Transaction Hour": transaction_hour
    }

    data = pd.DataFrame([transaction])


    # -------------------------------------------------
    # Fraud Prediction
    # -------------------------------------------------

    fraud_probability = model.predict_proba(data)[0][1]

    risk_score = round(
        fraud_probability * 100,
        2
    )


    # -------------------------------------------------
    # Risk Level
    # -------------------------------------------------

    if risk_score >= 70:

        risk_level = "HIGH RISK"

    elif risk_score >= 40:

        risk_level = "MEDIUM RISK"

    else:

        risk_level = "LOW RISK"


    # -------------------------------------------------
    # Risk Assessment
    # -------------------------------------------------

    st.divider()

    st.header("🛡️ Risk Assessment")

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Fraud Probability",
            f"{risk_score}%"
        )


    with col2:

        st.metric(
            "Risk Score",
            f"{risk_score}/100"
        )


    with col3:

        st.metric(
            "Risk Level",
            risk_level
        )


    # -------------------------------------------------
    # Risk Decision
    # -------------------------------------------------

    st.subheader("Risk Decision")


    if risk_score >= 70:

        st.error(
            "🚨 HIGH RISK — Transaction should be reviewed."
        )

    elif risk_score >= 40:

        st.warning(
            "⚠️ MEDIUM RISK — Additional verification recommended."
        )

    else:

        st.success(
            "✅ LOW RISK — Transaction appears relatively safe."
        )


    # -------------------------------------------------
    # SHAP Explanation
    # -------------------------------------------------

    st.divider()

    st.header("🔎 Why is this transaction risky?")

    st.write(
        "SHAP explains which transaction features influenced "
        "the fraud prediction."
    )


    # Get pipeline components

    preprocessor = model.named_steps["preprocessor"]

    rf_model = model.named_steps["model"]


    # Transform input

    transformed_data = preprocessor.transform(data)


    if hasattr(
        transformed_data,
        "toarray"
    ):

        transformed_data = transformed_data.toarray()


    # Feature names

    feature_names = (
        preprocessor.get_feature_names_out()
    )


    # SHAP Tree Explainer

    explainer = shap.TreeExplainer(
        rf_model
    )

    shap_values = explainer.shap_values(
        transformed_data
    )


    # Get fraud-class SHAP values

    if isinstance(
        shap_values,
        list
    ):

        fraud_shap_values = (
            shap_values[1][0]
        )

    else:

        fraud_shap_values = (
            shap_values[0, :, 1]
        )


    # Explanation table

    explanation = pd.DataFrame({

        "Feature": feature_names,

        "SHAP Value": fraud_shap_values
    })


    explanation["Impact"] = (
        explanation["SHAP Value"].abs()
    )


    # Top 5 factors

    explanation = (
        explanation
        .sort_values(
            "Impact",
            ascending=False
        )
        .head(5)
    )


    # Clean feature names

    explanation["Feature"] = (

        explanation["Feature"]

        .str.replace(
            "remainder__",
            "",
            regex=False
        )

        .str.replace(
            "categorical__",
            "",
            regex=False
        )

        .str.replace(
            "_",
            " ",
            regex=False
        )
    )


    # -------------------------------------------------
    # SHAP Chart
    # -------------------------------------------------

    st.subheader("📊 Top Risk Factors")


    chart_data = (
        explanation
        .sort_values(
            "SHAP Value",
            ascending=True
        )
    )


    fig, ax = plt.subplots(
        figsize=(9, 4)
    )


    ax.barh(
        chart_data["Feature"],
        chart_data["SHAP Value"]
    )

    ax.axvline(0)

    ax.set_xlabel(
        "SHAP Impact"
    )

    ax.set_ylabel(
        "Transaction Feature"
    )

    ax.set_title(
        "Features Influencing Fraud Risk"
    )


    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # -------------------------------------------------
    # SHAP Details
    # -------------------------------------------------

    st.subheader("📋 SHAP Details")


    st.dataframe(

        explanation[
            [
                "Feature",
                "SHAP Value"
            ]
        ],

        width="stretch"
    )


    # -------------------------------------------------
    # Simple Explanation
    # -------------------------------------------------

    st.subheader(
        "📌 Risk Factor Explanation"
    )


    for _, row in explanation.iterrows():

        feature = row["Feature"]

        shap_value = row["SHAP Value"]


        if shap_value > 0:

            st.write(
                f"🔴 **{feature}** increased the fraud risk."
            )

        else:

            st.write(
                f"🟢 **{feature}** decreased the fraud risk."
            )


    # -------------------------------------------------
    # Business Impact
    # -------------------------------------------------

    st.divider()

    st.header("💰 Business Impact")

    st.write(
        "Estimate the operational cost of manually reviewing "
        "a risky transaction."
    )


    review_cost = st.number_input(

        "Cost of Reviewing One Risky Transaction (₹)",

        min_value=0.0,

        value=100.0,

        step=10.0
    )


    # Estimate review cost

    if risk_score >= 70:

        estimated_cost = review_cost

        business_impact = "High"

    elif risk_score >= 40:

        estimated_cost = review_cost * 0.5

        business_impact = "Medium"

    else:

        estimated_cost = 0.0

        business_impact = "Low"


    col1, col2 = st.columns(2)


    with col1:

        st.metric(

            "Estimated Review Cost",

            f"₹{estimated_cost:.2f}"
        )


    with col2:

        st.metric(

            "Business Impact",

            business_impact
        )


    # Business recommendation

    if risk_score >= 70:

        st.warning(

            "💡 Recommendation: Send this transaction for manual "
            "review before approving it."
        )

    elif risk_score >= 40:

        st.info(

            "💡 Recommendation: Consider additional verification "
            "before completing the transaction."
        )

    else:

        st.success(

            "💡 Recommendation: Transaction can proceed with "
            "normal checks."
        )


    # -------------------------------------------------
    # False Positive vs Missed Fraud Cost
    # -------------------------------------------------

    st.divider()

    st.header(
        "⚖️ False Positive vs Missed Fraud Cost"
    )

    st.write(

        "This analysis shows the business trade-off between "
        "incorrectly flagging genuine customers and missing "
        "fraudulent transactions at different decision thresholds."
    )


    col1, col2 = st.columns(2)


    with col1:

        decision_threshold = st.slider(

            "Decision Threshold",

            min_value=0.10,

            max_value=0.90,

            value=0.50,

            step=0.05
        )


    with col2:

        missed_fraud_cost = st.number_input(

            "Estimated Cost of One Missed Fraud (₹)",

            min_value=0.0,

            value=500.0,

            step=50.0
        )


    false_positive_cost = review_cost


    # -------------------------------------------------
    # Load Evaluation Data
    # -------------------------------------------------

    @st.cache_data
    def load_evaluation_data():

        df = pd.read_csv(
            "data/risk_data.csv"
        )


        feature_columns = [

            "Transaction Amount",

            "Payment Method",

            "Product Category",

            "Quantity",

            "Customer Age",

            "Device Used",

            "Account Age Days",

            "Transaction Hour"
        ]


        X = df[feature_columns]

        y = df["Is Fraudulent"]


        return X, y


    X_eval, y_eval = (
        load_evaluation_data()
    )


    # -------------------------------------------------
    # Evaluation Sample
    # -------------------------------------------------

    sample_size = min(
        20000,
        len(X_eval)
    )


    genuine_data = X_eval[
        y_eval == 0
    ]


    fraud_data = X_eval[
        y_eval == 1
    ]


    genuine_sample_size = min(

        int(sample_size * 0.95),

        len(genuine_data)
    )


    fraud_sample_size = min(

        int(sample_size * 0.05),

        len(fraud_data)
    )


    genuine_sample = genuine_data.sample(

        n=genuine_sample_size,

        random_state=42
    )


    fraud_sample = fraud_data.sample(

        n=fraud_sample_size,

        random_state=42
    )


    evaluation_data = pd.concat(

        [
            genuine_sample,
            fraud_sample
        ]
    )


    evaluation_labels = y_eval.loc[
        evaluation_data.index
    ]


    # -------------------------------------------------
    # Model Probabilities
    # -------------------------------------------------

    evaluation_probabilities = (
        model.predict_proba(
            evaluation_data
        )[:, 1]
    )


    # -------------------------------------------------
    # Apply Selected Threshold
    # -------------------------------------------------

    evaluation_predictions = (

        evaluation_probabilities
        >= decision_threshold

    ).astype(int)


    # -------------------------------------------------
    # Dynamic Confusion Matrix
    # -------------------------------------------------

    tn = int(

        (
            (evaluation_labels == 0)
            &
            (evaluation_predictions == 0)
        ).sum()
    )


    fp = int(

        (
            (evaluation_labels == 0)
            &
            (evaluation_predictions == 1)
        ).sum()
    )


    fn = int(

        (
            (evaluation_labels == 1)
            &
            (evaluation_predictions == 0)
        ).sum()
    )


    tp = int(

        (
            (evaluation_labels == 1)
            &
            (evaluation_predictions == 1)
        ).sum()
    )


    # -------------------------------------------------
    # Business Cost Calculation
    # -------------------------------------------------

    total_false_positive_cost = (

        fp
        *
        false_positive_cost
    )


    total_missed_fraud_cost = (

        fn
        *
        missed_fraud_cost
    )


    total_estimated_cost = (

        total_false_positive_cost
        +
        total_missed_fraud_cost
    )


    # -------------------------------------------------
    # Cost Trade-off
    # -------------------------------------------------

    st.subheader(
        "📊 Cost Trade-off"
    )


    cost_data = pd.DataFrame({

        "Cost Category": [

            "False Positive Cost",

            "Missed Fraud Cost"
        ],

        "Estimated Cost (₹)": [

            total_false_positive_cost,

            total_missed_fraud_cost
        ]
    })


    st.dataframe(

        cost_data,

        width="stretch",

        hide_index=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(

            "False Positives",

            f"{fp:,}"
        )


    with col2:

        st.metric(

            "Missed Frauds",

            f"{fn:,}"
        )


    with col3:

        st.metric(

            "Total Estimated Cost",

            f"₹{total_estimated_cost:,.2f}"
        )


    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------

    st.subheader(
        "🔢 Confusion Matrix"
    )


    confusion_matrix_data = pd.DataFrame(

        [
            [tn, fp],

            [fn, tp]
        ],

        index=[

            "Actual Genuine",

            "Actual Fraud"
        ],

        columns=[

            "Predicted Genuine",

            "Predicted Fraud"
        ]
    )


    st.dataframe(

        confusion_matrix_data,

        width="stretch"
    )


    # -------------------------------------------------
    # Cost Comparison Chart
    # -------------------------------------------------

    fig2, ax2 = plt.subplots(

        figsize=(9, 4)
    )


    ax2.bar(

        cost_data["Cost Category"],

        cost_data["Estimated Cost (₹)"]
    )


    ax2.set_ylabel(
        "Estimated Cost (₹)"
    )


    ax2.set_title(
        "False Positive Cost vs Missed Fraud Cost"
    )


    plt.tight_layout()

    st.pyplot(fig2)

    plt.close(fig2)


    # -------------------------------------------------
    # Final Explanation
    # -------------------------------------------------

    st.info(

        f"💡 At a decision threshold of "
        f"{decision_threshold:.2f}, the evaluation data produced "
        f"{fp:,} false positives and {fn:,} missed frauds. "
        f"The estimated combined business cost is "
        f"₹{total_estimated_cost:,.2f}."
    )