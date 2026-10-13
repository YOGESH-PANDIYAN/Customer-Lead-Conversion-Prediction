
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Step 1: Load dataset
data = pd.read_csv("data/leads_large.csv")

# Step 2: Normalize text columns
categorical_features = [
    "location",
    "property_type",
    "lead_source"
]

for column in categorical_features:
    data[column] = data[column].str.strip().str.lower()

# Step 3: Separate inputs and target
X = data.drop(columns=["lead_id", "converted"])
y = data["converted"]

numerical_features = [
    "age",
    "budget",
    "calls",
    "follow_ups",
    "site_visit",
    "response_time_hours"
]

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 5: Define preprocessing
def make_preprocessor():
    return ColumnTransformer(
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

# Step 6: Define the models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
}

results = []
trained_models = {}

# Step 7: Train and evaluate
for name, classifier in models.items():

    pipeline = Pipeline([
        ("preprocessing", make_preprocessor()),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test, probabilities
        )
    })

    trained_models[name] = pipeline

    print(f"\n{name}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

# Step 8: Compare results
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# Step 9: Select best model by ROC-AUC
best_row = results_df.loc[results_df["ROC-AUC"].idxmax()]
best_name = best_row["Model"]

print(f"\nBest model by test ROC-AUC: {best_name}")

# Step 10: Save model
joblib.dump(
    trained_models[best_name],
    "models/best_lead_model.pkl"
)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nModel and comparison results saved!")
