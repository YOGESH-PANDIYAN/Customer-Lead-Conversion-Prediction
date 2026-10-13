
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Step 1: Load dataset
data = pd.read_csv("data/leads.csv")

# Step 2: Separate features and target
X = data.drop(columns=["lead_id", "converted"])
y = data["converted"]

# Step 3: Identify feature types
categorical_features = [
    "location",
    "property_type",
    "lead_source"
]

numerical_features = [
    "age",
    "budget",
    "calls",
    "follow_ups",
    "site_visit",
    "response_time_hours"
]

# Step 4: Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)

# Step 5: Create model pipeline
model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# Step 6: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 7: Train model
model.fit(X_train, y_train)

print("Model training completed!")

# Step 8: Predict on test data
y_pred = model.predict(X_test)

# Step 9: Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# Step 10: Save trained model
joblib.dump(model, "models/lead_conversion_model.pkl")

print("\nModel saved successfully!")
