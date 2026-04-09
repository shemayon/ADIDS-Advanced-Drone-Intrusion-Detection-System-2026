# train_model.py

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def train():
    data_file = "A-DIDS/data/drone_dataset.parquet"
    if not os.path.exists(data_file):
        print(f"[ERROR] {data_file} not found. Run data_pipeline.py first.")
        return

    print(f"[INFO] Loading dataset: {data_file}")
    df = pd.read_parquet(data_file)

    X = df.drop("label", axis=1)
    y = df["label"]

    print("[INFO] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[INFO] Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("[INFO] Evaluating model...")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    model_path = "A-DIDS/models/model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[INFO] Model saved as {model_path}")

if __name__ == "__main__":
    train()
