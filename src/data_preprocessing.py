import pandas as pd
from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv("data/risk_data.csv")

# Select useful columns
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

# Convert categorical columns into numerical columns
X = pd.get_dummies(
    X,
    columns=["Payment Method", "Product Category", "Device Used"],
    drop_first=True
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
print("Preprocessing completed successfully!")