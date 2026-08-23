# 🛡️ RiskLens

## AI-Powered Transaction Fraud Risk Manager

RiskLens is an explainable machine learning application that analyzes online transactions and estimates the probability of fraud.

The system combines **Machine Learning, SHAP Explainability, and Streamlit** to provide a practical fraud-risk management dashboard.

---

## 🎯 Project Objective

The goal of RiskLens is to help identify potentially fraudulent transactions before they are approved.

Instead of only predicting whether a transaction is fraudulent, RiskLens also explains **why** the transaction received a particular risk score.

---

## ✨ Key Features

- 🔍 Fraud probability prediction
- 📊 Risk Score from 0–100
- 🚦 HIGH / MEDIUM / LOW risk classification
- 🔎 Explainable AI using SHAP
- 📊 Visual SHAP risk-factor chart
- 💰 Business impact estimation
- 💡 Automated risk recommendations
- 🖥️ Interactive Streamlit dashboard

---

## 🤖 Machine Learning

RiskLens uses a machine learning pipeline to process transaction data and predict fraud probability.

### Input Features

The model uses transaction-level information such as:

- Transaction Amount
- Payment Method
- Product Category
- Quantity
- Customer Age
- Device Used
- Account Age
- Transaction Hour

The preprocessing pipeline handles categorical and numerical features before sending them to the trained model.

---

## 🔎 Explainable AI

RiskLens uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions.

For each transaction, the application identifies the top factors influencing the fraud prediction.

Example:

```text
Account Age Days       → Increased risk
Transaction Amount     → Increased risk
Transaction Hour       → Increased risk
Customer Age           → Decreased risk
Product Category       → Decreased risk