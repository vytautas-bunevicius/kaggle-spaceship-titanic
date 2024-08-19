"""Unit tests for the data_preprocessing_utils module."""

import pytest
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from spaceship_titanic.data_preprocessing_utils import (
    detect_anomalies_iqr,
    flag_anomalies,
    calculate_cramers_v,
    handle_missing_values,
    simple_imputation,
    engineer_spaceship_features,
    confidence_interval,
    create_pipeline,
)


@pytest.fixture(name="sample_df")
def fixture_sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 100],
            "B": [10, 20, 30, 40, 50],
            "C": ["x", "y", "z", "x", "y"],
        }
    )


@pytest.fixture(name="sample_spaceship_df")
def fixture_sample_spaceship_df():
    """Create a sample spaceship DataFrame for testing."""
    return pd.DataFrame(
        {
            "PassengerId": ["0001_01", "0002_01", "0003_01"],
            "HomePlanet": ["Earth", "Mars", "Europa"],
            "CryoSleep": [True, False, True],
            "Cabin": ["B/0/P", "F/1/S", "A/2/P"],
            "Age": [20, 35, 60],
            "RoomService": [100, 0, 50],
            "FoodCourt": [200, 50, 0],
            "ShoppingMall": [300, 100, 0],
            "Spa": [400, 150, 0],
            "VRDeck": [500, 200, 0],
        }
    )


def test_detect_anomalies_iqr(sample_df):
    """Test the detect_anomalies_iqr function."""
    anomalies = detect_anomalies_iqr(sample_df, ["A", "B"])
    assert len(anomalies) == 1
    assert anomalies["A"].values[0] == 100


def test_flag_anomalies(sample_df):
    """Test the flag_anomalies function."""
    flags = flag_anomalies(sample_df, ["A", "B"])
    assert flags.sum() == 1
    assert flags.iloc[-1]


def test_calculate_cramers_v(sample_df):
    """Test the calculate_cramers_v function."""
    cramer_v = calculate_cramers_v(sample_df["A"], sample_df["C"])
    assert 0 <= cramer_v <= 1


def test_handle_missing_values(capsys):
    """Test the handle_missing_values function."""
    df = pd.DataFrame(
        {
            "A": [1, 2, np.nan, 4],
            "B": [np.nan, 2, 3, 4],
            "C": [1, np.nan, np.nan, 4],
        }
    )
    cleaned_df = handle_missing_values(df, threshold=0.6)  # Increased threshold

    # Check that no columns were dropped
    assert set(cleaned_df.columns) == set(df.columns)

    # Check that rows with missing values were removed
    assert len(cleaned_df) == 1  # Only the last row should remain
    assert cleaned_df.iloc[0].tolist() == [4.0, 4.0, 4.0]

    # Check that the function prints the correct information
    captured = capsys.readouterr()
    assert "Columns dropped due to >60.0% missing values: []" in captured.out
    assert "Rows removed due to missing values: 3" in captured.out


def test_simple_imputation():
    """Test the simple_imputation function."""
    train_df = pd.DataFrame(
        {"A": [1, 2, np.nan, 4], "B": ["x", "y", "z", np.nan]}
    )
    test_df = pd.DataFrame({"A": [5, np.nan, 7], "B": [np.nan, "y", "z"]})
    imputed_train, imputed_test = simple_imputation(train_df, test_df)
    assert imputed_train["A"].isnull().sum() == 0
    assert imputed_test["A"].isnull().sum() == 0
    assert imputed_train["B"].isnull().sum() == 0
    assert imputed_test["B"].isnull().sum() == 0


def test_engineer_spaceship_features(sample_spaceship_df):
    """Test the engineer_spaceship_features function."""
    engineered_df = engineer_spaceship_features(sample_spaceship_df)
    assert "TotalSpending" in engineered_df.columns
    assert "CabinDeck" in engineered_df.columns
    assert "CabinNumber" in engineered_df.columns
    assert "CabinSide" in engineered_df.columns
    assert "GroupSize" in engineered_df.columns
    assert "AgeGroup" in engineered_df.columns
    assert "HomePlanetCryoSleep" in engineered_df.columns


def test_confidence_interval():
    """Test the confidence_interval function."""
    data = [1, 2, 3, 4, 5]
    mean, lower, upper = confidence_interval(data)
    assert lower < mean < upper


def test_create_pipeline():
    """Test the create_pipeline function."""
    preprocessor = SimpleImputer()
    model = LogisticRegression()
    pipeline = create_pipeline(preprocessor, model)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "preprocessor"
    assert pipeline.steps[1][0] == "classifier"


if __name__ == "__main__":
    pytest.main()
