
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load data
data = pd.read_csv("data/leads.csv")

print("Dataset Shape:", data.shape)

# Step 2: Check data quality
print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:", data.duplicated().sum())

# Step 3: Separate features and target
X = data.drop(columns=["lead_id", "converted"])
y = data["converted"]

# Step 4: Demonstrate categorical encoding
categorical_columns = [
    "location",
    "property_type",
    "lead_source"
]

X_encoded = pd.get_dummies(
    X,
    columns=categorical_columns,
    dtype=int
)

print("\nEncoded Dataset:")
print(X_encoded.head())

print("\nEncoded Shape:", X_encoded.shape)

# Step 5: Split original features before model preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Targets:", y_train.shape)
print("Testing Targets:", y_test.shape)

print("\nStep 5 completed successfully!")
