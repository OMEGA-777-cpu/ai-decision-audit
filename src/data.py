"""
Data loading for the AI Decision Audit project.

Dataset: Wisconsin Diagnostic Breast Cancer (WDBC), shipped with scikit-learn.
569 samples, 30 numeric features computed from digitized images of a breast
mass fine-needle aspirate. Binary target: malignant (1) / benign (0).

Chosen deliberately over a toy dataset (iris, digits) because the target
variable maps onto a real predictive-AI-informs-decision-making-AI scenario:
a model flags a case, and a human decides what happens next.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd


def load_data(test_size: float = 0.2, random_state: int = 42):
    raw = load_breast_cancer(as_frame=True)
    X = raw.data
    y = raw.target  # 0 = malignant, 1 = benign (sklearn's own encoding)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feats = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features: {len(feats)}")
    print(f"Train class balance:\n{y_train.value_counts(normalize=True)}")
