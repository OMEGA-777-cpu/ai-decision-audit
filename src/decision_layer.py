"""
Predictive AI vs. Decision-Making AI (IBM SkillsBuild: "AI Forms and Functions").

A predictive model only outputs a probability. A decision-making AI system
takes that probability and a policy, and outputs an ACTION. This module is
that policy layer, kept deliberately separate from the model so it can be
unit-tested on its own -- the thresholds are a business/clinical policy
choice, not a modelling choice, and auditing that policy is exactly the kind
of thing a "software testing" mindset should catch before deployment.

Malignancy probability convention: p = P(malignant).
sklearn's breast cancer target encodes 0 = malignant, 1 = benign, so
p_malignant = 1 - P(class 1).
"""

AUTO_CLEAR = "AUTO_CLEAR"
FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
URGENT_REVIEW = "URGENT_REVIEW"


def decide(p_malignant: float, low: float = 0.10, high: float = 0.50) -> str:
    """Map a malignancy probability to a discrete clinical-workflow action.

    Thresholds are intentionally asymmetric: because a false "auto-clear" is
    far more costly than an unnecessary review, the low threshold is kept
    conservative rather than set at the naive 0.5 midpoint.
    """
    if not 0.0 <= p_malignant <= 1.0:
        raise ValueError(f"p_malignant must be in [0, 1], got {p_malignant}")
    if p_malignant < low:
        return AUTO_CLEAR
    if p_malignant < high:
        return FLAG_FOR_REVIEW
    return URGENT_REVIEW


def decide_batch(probs_malignant, low: float = 0.10, high: float = 0.50):
    return [decide(p, low, high) for p in probs_malignant]
