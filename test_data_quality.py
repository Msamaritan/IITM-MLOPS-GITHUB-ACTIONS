import os
import pandas as pd
import pytest

DATA_PATH = "dvc_data/iris_data.csv"

@pytest.fixture
def load_dataset():
    """Fixture to load the dataset for testing"""
    assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}. Ensure 'dvc pull' has been executed."
    df = pd.read_csv(DATA_PATH)
    return df

def test_expected_schema(load_dataset):
    """Validate expected schema test"""
    df = load_dataset
    expected_columns = {
        "sepal_length", 
        "sepal_width", 
        "petal_length", 
        "petal_width", 
        "species"
    }
    assert expected_columns.issubset(set(df.columns)), f"Missing required columns. Found: {list(df.columns)}"

def test_no_missing_values(load_dataset):
    """ Test for missing/null values across all columns"""
    df = load_dataset
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    
    missing_counts = df[feature_cols].isnull().sum()
    assert missing_counts.sum() == 0, f"Found missing values in dataset:\n{missing_counts[missing_counts > 0]}"

def test_correct_feature_types(load_dataset):
    """Test fir correct feature types"""
    df = load_dataset
    numerical_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    
    for col in numerical_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' should be numeric, got {df[col].dtype}"
    
    assert pd.api.types.is_string_dtype(df["species"]) or pd.api.types.is_object_dtype(df["species"]), \
        f"Species column should be string/object, got {df['species'].dtype}"

def test_reasonable_value_ranges(load_dataset):
    """Check for reasonable physical value ranges for Iris features"""
    df = load_dataset
    
    # Domain-specific physical limits for Iris flower dimensions (in cm)
    assert df["sepal_length"].between(4.0, 9.0).all(), "sepal_length contains values outside expected range (4.0 - 9.0)"
    assert df["sepal_width"].between(1.5, 5.0).all(), "sepal_width contains values outside expected range (1.5 - 5.0)"
    assert df["petal_length"].between(0.5, 8.0).all(), "petal_length contains values outside expected range (0.5 - 8.0)"
    assert df["petal_width"].between(0.0, 3.5).all(), "petal_width contains values outside expected range (0.0 - 3.5)"
    
    # Validate target class categories
    expected_species = {"setosa", "versicolor", "virginica"}
    actual_species = set(df["species"].unique())
    assert actual_species.issubset(expected_species), f"Unexpected species categories found: {actual_species - expected_species}"