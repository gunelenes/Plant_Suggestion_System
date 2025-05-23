# tests/test_model_artifacts.py

import os
import pytest
import joblib

@pytest.fixture(scope="session")
def model_paths():
    """
    Ensure that model artifacts exist before running dependent tests.
    """
    model_pkl = os.path.join("models", "feedback_model.pkl")
    vec_pkl   = os.path.join("models", "feedback_vec.pkl")
    report    = "last_feedback_model_report.txt"
    return {"model": model_pkl, "vec": vec_pkl, "report": report}

def test_model_artifacts_exist(model_paths):
    """Check that the model, vectorizer and report files are present."""
    assert os.path.isfile(model_paths["model"]), f"Model file not found: {model_paths['model']}"
    assert os.path.isfile(model_paths["vec"]),   f"Vectorizer file not found: {model_paths['vec']}"
    assert os.path.isfile(model_paths["report"]), f"Report file not found: {model_paths['report']}"

def test_model_loadable_and_methods(model_paths):
    """Load the model and verify it has necessary methods."""
    clf = joblib.load(model_paths["model"])
    assert hasattr(clf, "predict"), "Loaded model missing `predict` method"
    assert hasattr(clf, "predict_proba"), "Loaded model missing `predict_proba` method"

def test_vectorizer_loadable_and_methods(model_paths):
    """Load the vectorizer and verify it has `transform` method."""
    vec = joblib.load(model_paths["vec"])
    assert hasattr(vec, "transform"), "Loaded vectorizer missing `transform` method"

def test_report_contains_metrics(model_paths):
    """Verify that the report includes precision, recall, and accuracy."""
    text = open(model_paths["report"], encoding="utf-8").read().lower()
    assert "precision" in text, "Report missing 'precision'"
    assert "recall"    in text, "Report missing 'recall'"
    assert "accuracy"  in text, "Report missing 'accuracy'"

def test_model_predict_proba_shape(model_paths):
    """
    Feed a minimal valid input to the vectorizer and model,
    and assert predict_proba returns shape (n_samples, 2).
    """
    vec = joblib.load(model_paths["vec"])
    clf = joblib.load(model_paths["model"])
    # Minimal sample: must match features used in training
    # Example keys: replace with actual feature names from your vectorizer
    sample = [{
        "area_size": "Small",
        "sunlight_need": "Shade",
        "environment_type": "Indoor",
        "climate_type": "All seasons",
        "watering_frequency": "Weekly",
        "fertilizer_frequency": "Never needed",
        "pesticide_frequency": "Never needed",
        "has_pet": "No",
        "has_child": "No",
        "suggested_plant": "Aloe Vera"
    }]
    X = vec.transform(sample)
    probs = clf.predict_proba(X)
    assert probs.shape == (1, 2), f"predict_proba output shape was {probs.shape}, expected (1, 2)"
