"""
pytest suite. Run with: pytest -v

Mirrors the CI-gated testing pattern used in ml-model-test-framework:
tests are the contract, not an afterthought.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.decision_layer import decide, decide_batch, AUTO_CLEAR, FLAG_FOR_REVIEW, URGENT_REVIEW
from src.data import load_data
from src.models import build_models, train_and_evaluate


# ---- Decision layer: boundary and contract tests ----

def test_decide_low_probability_auto_clears():
    assert decide(0.01) == AUTO_CLEAR


def test_decide_high_probability_urgent():
    assert decide(0.9) == URGENT_REVIEW


def test_decide_mid_probability_flagged():
    assert decide(0.3) == FLAG_FOR_REVIEW


def test_decide_boundary_low_is_inclusive_of_flag():
    # exactly at the low threshold should NOT auto-clear
    assert decide(0.10) == FLAG_FOR_REVIEW


def test_decide_boundary_high_is_urgent():
    # exactly at the high threshold should escalate, not just flag
    assert decide(0.50) == URGENT_REVIEW


def test_decide_rejects_out_of_range_probability():
    with pytest.raises(ValueError):
        decide(1.5)
    with pytest.raises(ValueError):
        decide(-0.1)


def test_decide_batch_matches_single_calls():
    probs = [0.01, 0.3, 0.9]
    assert decide_batch(probs) == [decide(p) for p in probs]


# ---- Data contract tests ----

def test_data_split_has_no_overlap():
    X_train, X_test, y_train, y_test, _ = load_data()
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_data_split_is_stratified_reasonably():
    X_train, X_test, y_train, y_test, _ = load_data()
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.05


# ---- Model quality gate: this is the test that should fail CI if a future
# change (new features, different model) silently tanks accuracy. ----

def test_all_models_beat_majority_baseline():
    X_train, X_test, y_train, y_test, _ = load_data()
    baseline = max(y_test.mean(), 1 - y_test.mean())
    models = build_models()
    results = train_and_evaluate(models, X_train, y_train, X_test, y_test)
    for name, r in results.items():
        assert r["accuracy"] > baseline, f"{name} did not beat the majority-class baseline"


def test_neural_network_recall_meets_clinical_floor():
    """Recall on the malignant class is the metric that matters most here --
    a missed malignant case is far worse than a false alarm. Fail CI if it
    drops below a floor a reviewer would actually accept."""
    X_train, X_test, y_train, y_test, _ = load_data()
    models = build_models()
    results = train_and_evaluate(models, X_train, y_train, X_test, y_test)
    assert results["neural_network"]["recall"] > 0.85
