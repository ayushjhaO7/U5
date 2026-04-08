import pytest
import os
import pandas as pd
import joblib
from src.data.data_preparation import preprocess_crime_data, TOP_CRIMES
from src.model.model_training import train_kmeans_model

def test_data_preparation():
    input_file = "test_raw_crime.csv"
    output_file = "test_cleaned_crime.csv"

    df = pd.DataFrame({
        'STATE/UT': ['A', 'A', 'B', 'B', 'C', 'C'],
        'crime_type': ['THEFT', 'AUTO THEFT', 'THEFT', 'BURGLARY', 'THEFT', 'OTHER THEFT'],
        'crime_count': [100, 200, 300, 400, 500, 600]
    })
    df.to_csv(input_file, index=False)

    preprocess_crime_data(input_file, output_file)

    assert os.path.exists(output_file)
    df_result = pd.read_csv(output_file)
    assert 'STATE/UT' in df_result.columns
    assert 'THEFT' in df_result.columns
    assert len(df_result) == 3

    os.remove(input_file)
    os.remove(output_file)

def test_model_training():
    data_file = "test_train_crime.csv"
    model_file = "test_model_crime.pkl"

    df = pd.DataFrame({
        'STATE/UT': ['State1', 'State2', 'State3'],
        'THEFT': [100, 1000, 5000],
        'HURT/GREVIOUS HURT': [50, 500, 2500],
        'OTHER THEFT': [10, 100, 1000],
        'AUTO THEFT': [5, 50, 250],
        'BURGLARY': [1, 10, 100]
    })
    df.to_csv(data_file, index=False)

    train_kmeans_model(data_file, model_file)

    assert os.path.exists(model_file)
    assert os.path.exists("scaler.pkl")
    assert os.path.exists("model_meta.joblib")

    model = joblib.load(model_file)
    assert hasattr(model, "predict")

    meta = joblib.load("model_meta.joblib")
    assert "risk_mapping" in meta
    assert meta["risk_mapping"] is not None

    os.remove(data_file)
    os.remove(model_file)
