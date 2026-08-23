# 🛡️ RiskLens

### AI-Powered Transaction Fraud Risk Manager

RiskLens is an explainable machine learning application that analyzes transaction data and estimates the probability of fraud.

It combines machine learning, SHAP explainability, and Streamlit to provide an interactive fraud-risk assessment dashboard.

---

## 🚀 Overview

Fraud detection systems often focus only on predicting whether a transaction is fraudulent.

RiskLens goes one step further by providing:

- Fraud probability
- Risk score
- Risk classification
- Explainable AI using SHAP
- Top transaction risk factors
- Business impact estimation
- Risk recommendations

The goal is to make fraud predictions easier to understand and useful for risk-management decisions.

---

## ✨ Key Features

- 🔍 **Fraud Prediction** — Estimates the probability that a transaction is fraudulent.
- 📊 **Risk Score** — Converts fraud probability into a 0–100 risk score.
- 🚦 **Risk Classification** — Categorizes transactions as Low, Medium, or High risk.
- 🔎 **SHAP Explainability** — Explains which features influenced the prediction.
- 📈 **Risk Factor Visualization** — Shows the most influential transaction features.
- 💰 **Business Impact Analysis** — Estimates potential operational review impact.
- 🖥️ **Interactive Dashboard** — Built using Streamlit.

---

## 🧠 Machine Learning Workflow

```text
Transaction Input
       ↓
Data Preprocessing
       ↓
Feature Transformation
       ↓
Machine Learning Model
       ↓
Fraud Probability
       ↓
Risk Score
       ↓
Risk Classification
       ↓
SHAP Explanation
       ↓
Business Risk Decision
---

## 🔎 Explainable AI

RiskLens uses **SHAP (SHapley Additive exPlanations)** to identify the transaction features that contribute most strongly to a prediction.

This helps users understand **why** a transaction received a particular risk score instead of treating the model as a black box.

---

## 🖥️ Dashboard Screenshots

### Main Dashboard

![RiskLens Dashboard](assets/screenshots/dashboard.png)

### Risk Assessment

![Risk Assessment](assets/screenshots/risk-assessment.png)

### SHAP Risk Analysis

![SHAP Analysis](assets/screenshots/shap-analysis.png)

### Business Impact

![Business Impact](assets/screenshots/business-impact.png)

### Risk Decision

![Risk Decision](assets/screenshots/risk-decision.png)

---

## 🗂️ Project Structure

```text
risklens-project/
│
├── assets/
│   └── screenshots/
│       ├── dashboard.png
│       ├── risk-assessment.png
│       ├── shap-analysis.png
│       ├── business-impact.png
│       └── risk-decision.png
│
├── data/
├── models/
├── notebooks/
├── src/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md