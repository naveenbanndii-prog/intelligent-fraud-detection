"""
Intelligent Fraud Detection - Simple Base Project
-------------------------------------------------
Pipeline:
  1. Load data (real CSV if available, otherwise synthetic data)
  2. Preprocess (split, scale)
  3. Train models:
       - Logistic Regression   (simple baseline)
       - Random Forest         (supervised, handles non-linear patterns)
       - Isolation Forest      (unsupervised anomaly detection)
  4. Evaluate (precision, recall, F1, ROC-AUC, confusion matrix)
  5. Save best model and predict on new transactions

Install:  pip install pandas numpy scikit-learn joblib
Run:      python fraud_detection.py
          python fraud_detection.py --csv creditcard.csv   (real data, column 'Class' = label)
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MODEL_PATH = "fraud_model.joblib"


# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------
def generate_synthetic_data(n_samples=20000, fraud_ratio=0.02, seed=RANDOM_STATE):
    """Create a fake transaction dataset with realistic fraud patterns."""
    rng = np.random.default_rng(seed)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate: small/medium amounts, daytime, home country, usual device
    legit = pd.DataFrame({
        "amount": rng.lognormal(mean=3.5, sigma=0.8, size=n_legit),
        "hour": rng.normal(14, 4, n_legit).clip(0, 23).astype(int),
        "distance_from_home_km": rng.exponential(10, n_legit),
        "transactions_last_24h": rng.poisson(2, n_legit),
        "is_foreign": rng.binomial(1, 0.03, n_legit),
        "is_new_device": rng.binomial(1, 0.05, n_legit),
        "account_age_days": rng.integers(60, 3000, n_legit),
        "Class": 0,
    })

    # Fraud: larger amounts, night-time, far away, bursts, new devices
    fraud = pd.DataFrame({
        "amount": rng.lognormal(mean=5.2, sigma=1.0, size=n_fraud),
        "hour": rng.choice([0, 1, 2, 3, 4, 22, 23], n_fraud),
        "distance_from_home_km": rng.exponential(200, n_fraud),
        "transactions_last_24h": rng.poisson(8, n_fraud),
        "is_foreign": rng.binomial(1, 0.6, n_fraud),
        "is_new_device": rng.binomial(1, 0.7, n_fraud),
        "account_age_days": rng.integers(1, 400, n_fraud),
        "Class": 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_data(csv_path=None):
    if csv_path and os.path.exists(csv_path):
        print(f"Loading real dataset: {csv_path}")
        return pd.read_csv(csv_path)
    print("No CSV supplied -> generating synthetic transaction data")
    return generate_synthetic_data()


# ----------------------------------------------------------------------
# 2. PREPROCESSING
# ----------------------------------------------------------------------
def preprocess(df, target="Class"):
    df = df.drop_duplicates().dropna()
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler, list(X.columns)


# ----------------------------------------------------------------------
# 3. MODELS
# ----------------------------------------------------------------------
def train_models(X_train, y_train):
    fraud_rate = y_train.mean()

    # class_weight="balanced" fixes the heavy class imbalance (fraud is rare)
    log_reg = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    ).fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ).fit(X_train, y_train)

    # Unsupervised: learns what "normal" looks like, flags outliers
    iso = IsolationForest(
        n_estimators=200,
        contamination=max(fraud_rate, 0.001),
        random_state=RANDOM_STATE,
    ).fit(X_train)

    return {"Logistic Regression": log_reg, "Random Forest": rf, "Isolation Forest": iso}


# ----------------------------------------------------------------------
# 4. EVALUATION
# ----------------------------------------------------------------------
def evaluate(name, model, X_test, y_test):
    if name == "Isolation Forest":
        # IsolationForest outputs -1 (anomaly) / 1 (normal)
        y_pred = (model.predict(X_test) == -1).astype(int)
        scores = -model.score_samples(X_test)  # higher = more anomalous
    else:
        y_pred = model.predict(X_test)
        scores = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, scores)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], digits=3))
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {auc:.4f} | F1 (fraud): {f1:.4f}")
    return f1, auc


# ----------------------------------------------------------------------
# 5. SAVE + PREDICT
# ----------------------------------------------------------------------
def save_model(model, scaler, columns, path=MODEL_PATH):
    joblib.dump({"model": model, "scaler": scaler, "columns": columns}, path)
    print(f"\nModel saved to {path}")


def predict_transaction(transaction: dict, path=MODEL_PATH, threshold=0.5):
    """Score a single transaction. Returns (is_fraud, probability)."""
    bundle = joblib.load(path)
    row = pd.DataFrame([transaction])[bundle["columns"]]
    prob = bundle["model"].predict_proba(bundle["scaler"].transform(row))[0, 1]
    return prob >= threshold, prob


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to a labelled CSV (label column: 'Class')")
    args = parser.parse_args()

    df = load_data(args.csv)
    print(f"Dataset: {df.shape[0]} rows | fraud rate: {df['Class'].mean():.2%}")

    X_train, X_test, y_train, y_test, scaler, columns = preprocess(df)
    models = train_models(X_train, y_train)

    results = {name: evaluate(name, m, X_test, y_test) for name, m in models.items()}

    # Pick best supervised model by F1 (Isolation Forest has no predict_proba)
    supervised = {k: v for k, v in results.items() if k != "Isolation Forest"}
    best = max(supervised, key=lambda k: supervised[k][0])
    print(f"\nBest supervised model: {best}")
    save_model(models[best], scaler, columns)

    # Demo: score two example transactions (only for the synthetic schema)
    if "distance_from_home_km" in columns:
        normal_tx = dict(amount=25.0, hour=13, distance_from_home_km=4,
                         transactions_last_24h=1, is_foreign=0,
                         is_new_device=0, account_age_days=900)
        suspicious_tx = dict(amount=1800.0, hour=2, distance_from_home_km=450,
                             transactions_last_24h=9, is_foreign=1,
                             is_new_device=1, account_age_days=12)
        for label, tx in [("Normal", normal_tx), ("Suspicious", suspicious_tx)]:
            flag, p = predict_transaction(tx)
            print(f"{label:<11} -> fraud={flag} (probability {p:.2%})")


if __name__ == "__main__":
    main()
