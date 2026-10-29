import os
import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

DATA_PATH = "dvc_data/iris_data.csv"
MODEL_PATH = "dvc_model/model.joblib"

# Quality thresholds for model validation
MIN_ACCURACY_THRESHOLD = 0.85
MIN_F1_THRESHOLD = 0.85

@pytest.fixture
def load_data_and_model():
    """Fixture to load evaluation data and the saved model artifact."""
    assert os.path.exists(DATA_PATH), f"Evaluation data not found at {DATA_PATH}. Ensure 'dvc pull' has been executed."
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}. Ensure 'dvc pull' has been executed."
    
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]
    
    return X, y, model

def test_model_artifact_loading(load_data_and_model):
    """Verify that the model artifact loads correctly and is functional"""
    _, _, model = load_data_and_model
    assert model is not None, "Loaded model object is None"
    assert hasattr(model, "predict"), "Loaded model object missing 'predict' method"

def test_model_inference_output_shape(load_data_and_model):
    """Test that inference output matches expected length and target class types"""
    X, y, model = load_data_and_model
    predictions = model.predict(X)
    
    assert len(predictions) == len(y), f"Prediction count ({len(predictions)}) does not match input count ({len(y)})"
    assert set(predictions).issubset({"setosa", "versicolor", "virginica"}), f"Unexpected species in predictions: {set(predictions)}"

def test_model_performance_thresholds(load_data_and_model):
    """Assert key classification metrics satisfy quality thresholds"""
    X, y_true, model = load_data_and_model
    y_pred = model.predict(X)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    
    print(f"\n--- Model Evaluation Metrics ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    assert acc >= MIN_ACCURACY_THRESHOLD, f"Model Accuracy ({acc:.4f}) is below threshold ({MIN_ACCURACY_THRESHOLD})"
    assert f1 >= MIN_F1_THRESHOLD, f"Model F1 Score ({f1:.4f}) is below threshold ({MIN_F1_THRESHOLD})"