
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "leads_large.csv"

df = pd.read_csv(DATA_PATH)

categorical_features = [
    "location",
    "property_type",
    "lead_source",
]

for col in categorical_features:
    df[col] = df[col].str.strip().str.lower()

X = df.drop(columns=["lead_id", "converted"])
y = df["converted"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

def make_pipeline(classifier):
    preprocessing = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features,
        ),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
    ),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_results = {}

print("\nCROSS-VALIDATION RESULTS")
print("-" * 45)

for name, classifier in models.items():
    pipeline = make_pipeline(classifier)

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
    )

    cv_results[name] = scores.mean()

    print(
        f"{name}: "
        f"{scores.mean():.4f} "
        f"(+/- {scores.std():.4f})"
    )

best_name = max(cv_results, key=cv_results.get)
best_model = make_pipeline(models[best_name])

best_model.fit(X_train, y_train)

predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]

print("\nSELECTED MODEL:", best_name)

print("\nHELD-OUT TEST RESULTS")
print("-" * 45)

print(
    "Accuracy:",
    round(accuracy_score(y_test, predictions), 4)
)

print(
    "Precision:",
    round(precision_score(
        y_test, predictions, zero_division=0
    ), 4)
)

print(
    "Recall:",
    round(recall_score(
        y_test, predictions, zero_division=0
    ), 4)
)

print(
    "F1 Score:",
    round(f1_score(
        y_test, predictions, zero_division=0
    ), 4)
)

print(
    "ROC-AUC:",
    round(roc_auc_score(
        y_test, probabilities
    ), 4)
)

print(
    "PR-AUC:",
    round(average_precision_score(
        y_test, probabilities
    ), 4)
)
