# Intelligent Fraud Detection

A simple machine learning project that detects fraudulent transactions.

## Features
- Synthetic transaction data generator (or load a real CSV)
- Preprocessing: de-duplication, stratified split, scaling
- Models: Logistic Regression, Random Forest, Isolation Forest (anomaly detection)
- Evaluation: precision, recall, F1, ROC-AUC, confusion matrix
- Saves the best model and scores new transactions

## Setup
```
pip install -r requirements.txt
```

## Run
```
python fraud_detection.py
```

Use a real labelled dataset (label column named `Class`, e.g. Kaggle Credit Card Fraud):
```
python fraud_detection.py --csv creditcard.csv
```

## Predict a single transaction
```python
from fraud_detection import predict_transaction

tx = dict(amount=1800.0, hour=2, distance_from_home_km=450,
          transactions_last_24h=9, is_foreign=1,
          is_new_device=1, account_age_days=12)
print(predict_transaction(tx))   # (True, 1.0)
```

## Project structure
```
fraud_detection_project/
├── fraud_detection.py   # main code
├── requirements.txt
└── README.md
```
