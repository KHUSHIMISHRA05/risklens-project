# 🛡️ RiskLens

## AI-Powered Transaction Fraud Risk Manager

RiskLens is an AI-powered fraud risk assessment system that predicts whether an e-commerce transaction may be fraudulent and provides an explainable risk score.

The system uses Machine Learning for fraud prediction and SHAP for explainable AI.

---

## 🎯 Project Objective

The main objective of RiskLens is to help identify potentially fraudulent transactions and assist risk teams in making better decisions.

The system provides:

- Fraud probability
- Risk score from 0–100
- Low, Medium, or High risk classification
- SHAP-based explanation
- Business impact estimation
- Recommended action

---

## 🚀 Key Features

### 🤖 Fraud Detection

A Machine Learning model predicts the probability that a transaction is fraudulent.

### 📊 Risk Score

The fraud probability is converted into a score between 0 and 100.

| Risk Score | Risk Level |
|---|---|
| 0–39 | 🟢 Low Risk |
| 40–69 | 🟡 Medium Risk |
| 70–100 | 🔴 High Risk |

### 🔎 Explainable AI

SHAP is used to explain which transaction features influenced the fraud prediction.

### 💰 Business Impact

The system estimates the operational cost of manually reviewing a risky transaction.

### 🌐 Interactive Dashboard

A Streamlit dashboard allows users to enter transaction details and receive an instant risk assessment.

---

## 🧠 System Workflow

```text
Transaction Details
        ↓
Data Preprocessing
        ↓
Machine Learning Model
        ↓
Fraud Probability
        ↓
Risk Score
        ↓
Risk Level
        ↓
SHAP Explanation
        ↓
Business Recommendation