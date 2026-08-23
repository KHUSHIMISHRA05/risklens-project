import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Load dataset
df = pd.read_csv("data/risk_data.csv")

# Features
features = [
    "Transaction Amount",
    "Payment Method",
    "Product Category",
    "Quantity",
    "Customer Age",
    "Device Used",
    "Account Age Days",
    "Transaction Hour"
]

target = "Is Fraudulent"

X = df[features]
y = df[target]

# Categorical and numerical columns
categorical_features = [
    "Payment Method",
    "Product Category",
    "Device Used"
]

numerical_features = [
    "Transaction Amount",
    "Quantity",
    "Customer Age",
    "Account Age Days",
    "Transaction Hour"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
print("Training model...")
pipeline.fit(X_train, y_train)

print("Model training completed!")

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(pipeline, "models/risk_model.pkl")

print("\nModel saved successfully!")