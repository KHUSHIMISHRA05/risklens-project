import joblib
import pandas as pd
import shap


# Load trained model
model = joblib.load("models/risk_model.pkl")


# Example transaction
transaction = {
    "Transaction Amount": 850.50,
    "Payment Method": "credit card",
    "Product Category": "electronics",
    "Quantity": 2,
    "Customer Age": 30,
    "Device Used": "mobile",
    "Account Age Days": 15,
    "Transaction Hour": 2
}

# Convert transaction to DataFrame
data = pd.DataFrame([transaction])


# Get prediction
prediction = model.predict(data)[0]
probability = model.predict_proba(data)[0][1]

print("Prediction:", "FRAUD" if prediction == 1 else "NORMAL")
print("Fraud Probability:", round(probability * 100, 2), "%")


# Get the trained Random Forest from the pipeline
rf_model = model.named_steps["model"]
preprocessor = model.named_steps["preprocessor"]


# Transform transaction using the same preprocessing
transformed_data = preprocessor.transform(data)

# Get feature names after encoding
feature_names = preprocessor.get_feature_names_out()

# Convert sparse matrix if necessary
if hasattr(transformed_data, "toarray"):
    transformed_data = transformed_data.toarray()


# SHAP explainer
explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(transformed_data)


# Get SHAP values for fraud class
if isinstance(shap_values, list):
    fraud_shap_values = shap_values[1][0]
else:
    fraud_shap_values = shap_values[0, :, 1]


# Show feature contributions
explanation = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Value": fraud_shap_values
})

explanation["Absolute Impact"] = explanation["SHAP Value"].abs()

explanation = explanation.sort_values(
    "Absolute Impact",
    ascending=False
)

print("\nTop Risk Factors:")
print(explanation.head(10).to_string(index=False))