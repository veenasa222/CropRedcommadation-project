# CropGuard - Random Forest Model Training

# Step 1: Import libraries
import os
import joblib
import pandas as pd
import numpy as np
from feature_engineering import add_features
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Step 2: Define input features
FEATURES = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]

TARGET = "label"

# Step 3: Load dataset
print("Loading dataset...")

data = pd.read_csv("data/Crop_recommendation.csv")

# Step 4: Clean dataset
data = data.drop_duplicates()
data = data.dropna(subset=FEATURES + [TARGET])

for column in FEATURES:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data = data.dropna(subset=FEATURES)

X = data[FEATURES]
y = data[TARGET]

print("Dataset rows:", len(data))
print("Number of crops:", y.nunique())

# Step 5: Feature engineering



# Step 6: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Step 7: Create Random Forest pipeline
# Feature engineering stays inside the pipeline to avoid leakage.
model = Pipeline([
    (
        "features",
        FunctionTransformer(add_features, validate=False)
    ),
    (
        "classifier",
        RandomForestClassifier(random_state=42)
    )
])

# Step 8: Tune Random Forest hyperparameters
parameters = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10, 15, None],
    "classifier__min_samples_split": [2, 5]
}

print("\nTraining and tuning Random Forest...")

search = GridSearchCV(
    model,
    parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("Best parameters:", search.best_params_)
print("Best cross-validation accuracy:", round(search.best_score_, 4))

# Step 9: Evaluate on unseen test data
predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, predictions)

print("\nTest accuracy:", round(test_accuracy * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# Step 10: Save the trained model
os.makedirs("models", exist_ok=True)

# Store feature names and training ranges for later validation
saved_model = {
    "model": best_model,
    "model_name": "Random Forest",
    "features": FEATURES,
    "input_ranges": {
        feature: {
            "min": float(data[feature].min()),
            "max": float(data[feature].max())
        }
        for feature in FEATURES
    },
    "confidence_threshold": 0.60,
    "best_parameters": search.best_params_
}

joblib.dump(saved_model, "models/crop_recommendation.pkl")

print("\nModel saved successfully!")
print("Location: backend/models/crop_recommendation.pkl")