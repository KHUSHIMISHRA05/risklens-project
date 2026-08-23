import joblib
import pandas as pd


# Load trained model
model = joblib.load("models/risk_model.pkl")


def calculate_risk(transaction):
    """
    Predict fraud probability and calculate risk score.
    """

    # Convert transaction into DataFrame
    data = pd.DataFrame([transaction])

    # Get fraud probability
    fraud_probability = model.predict_proba(data)[0][1]

    # Convert probability to 0-100 risk score
    risk_score = round(fraud_probability * 100, 2)

    # Determine risk level
    if risk_score >= 70:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "fraud_probability": round(fraud_probability, 4),
        "risk_score": risk_score,
        "risk_level": risk_level
    }


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


result = calculate_risk(transaction)

print("RiskLens Result")
print("----------------")
print("Fraud Probability:", result["fraud_probability"])
print("Risk Score:", result["risk_score"], "/ 100")
print("Risk Level:", result["risk_level"])