"""
Subgroup performance audit (IBM SkillsBuild: "The ethics of deep learning" /
"How machine learning helps AI improve").

Honest scope note (this matters -- don't oversell this section in the
report): the WDBC dataset carries no demographic attributes (no age, sex,
ethnicity, etc.), so a real protected-attribute fairness audit is NOT
possible on this data. What IS possible, and still a legitimate and commonly
used technique, is a subgroup performance audit: check whether the model's
error rate is stable across a feature-derived slice of the population,
here tumor size (mean_radius), split into quartiles.

If false-negative rate varied sharply by tumor size, that would be a real,
actionable finding (e.g. the model underperforming on borderline/small
tumors). This module measures that. It does not claim to be a demographic
bias audit, and the report says so explicitly.
"""
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score


def subgroup_audit(model, X_test: pd.DataFrame, y_test: pd.Series,
                    feature: str = "mean radius", n_bins: int = 4) -> pd.DataFrame:
    df = X_test.copy()
    df["_true"] = y_test.values
    df["_pred"] = model.predict(X_test)
    df["_bin"] = pd.qcut(df[feature], q=n_bins, labels=[f"Q{i+1}" for i in range(n_bins)])

    rows = []
    for bin_label, group in df.groupby("_bin", observed=True):
        # recall on the malignant class (0) matters most clinically:
        # a missed malignant case (false negative) is the costly error.
        malignant_mask = group["_true"] == 0
        n_malignant = malignant_mask.sum()
        if n_malignant > 0:
            recall_malignant = (
                (group.loc[malignant_mask, "_pred"] == 0).sum() / n_malignant
            )
        else:
            recall_malignant = float("nan")

        rows.append({
            "subgroup": bin_label,
            "n_samples": len(group),
            "n_malignant": int(n_malignant),
            "accuracy": round(accuracy_score(group["_true"], group["_pred"]), 4),
            "malignant_recall": round(recall_malignant, 4) if pd.notna(recall_malignant) else None,
        })
    return pd.DataFrame(rows)
