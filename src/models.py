"""
Trains three model families covered across the IBM SkillsBuild modules:

  - Logistic Regression   (Machine Learning: classical models)
  - Decision Tree         (Machine Learning: classical models)
  - MLP Neural Network    (Neural Networks and Deep Learning)

All three are wrapped behind a common interface (predict_proba) so the
decision layer and benchmarking code don't care which model produced a score.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def build_models(random_state: int = 42) -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
        ]),
        "decision_tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=4, random_state=random_state)),
        ]),
        "neural_network": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                max_iter=2000,
                random_state=random_state,
            )),
        ]),
    }


def train_and_evaluate(models: dict, X_train, y_train, X_test, y_test) -> dict:
    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, probs), 4),
            "fitted_model": pipe,
        }
    return results
