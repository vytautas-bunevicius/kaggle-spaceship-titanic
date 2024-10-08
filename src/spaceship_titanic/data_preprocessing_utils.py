"""
This module provides various functions for data preprocessing, anomaly
detection, feature engineering, and statistical analysis, specifically
tailored for machine learning workflows.

Functions included:
- `detect_anomalies_iqr`: Detects anomalies in multiple features using the
  Interquartile Range (IQR) method.
- `flag_anomalies`: Flags anomalies in specified features using the IQR
  method.
- `calculate_cramers_v`: Computes Cramer's V statistic for
  categorical-categorical association.
- `handle_missing_values`: Handles missing data by dropping columns or rows
  based on a threshold.
- `simple_imputation`: Performs simple imputation on missing values in
  training and testing datasets.
- `engineer_spaceship_features`: Performs feature engineering, particularly
  for a spaceship passenger dataset.
- `confidence_interval`: Calculates the confidence interval for a given
  dataset.
- `create_pipeline`: Creates a scikit-learn pipeline for preprocessing and
  modeling.

This module is intended for use in data preprocessing and feature engineering,
with functions designed to handle common tasks in machine learning pipelines,
such as anomaly detection, missing data handling, and feature engineering.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline


def detect_anomalies_iqr(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Detects anomalies in multiple features using the IQR method.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        features (List[str]): List of features to detect anomalies in.

    Returns:
        pd.DataFrame: DataFrame containing the anomalies for each feature.
    """
    anomalies_list = []

    for feature in features:
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in DataFrame.")
            continue

        if not np.issubdtype(df[feature].dtype, np.number):
            print(f"Feature '{feature}' is not numerical and will be skipped.")
            continue

        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        feature_anomalies = df[
            (df[feature] < lower_bound) | (df[feature] > upper_bound)
        ]
        if not feature_anomalies.empty:
            print(f"Anomalies detected in feature '{feature}':")
            print(feature_anomalies)
        else:
            print(f"No anomalies detected in feature '{feature}'.")
        anomalies_list.append(feature_anomalies)

    if anomalies_list:
        anomalies = (
            pd.concat(anomalies_list).drop_duplicates().reset_index(drop=True)
        )
        anomalies = anomalies[features]
    else:
        anomalies = pd.DataFrame(columns=features)

    return anomalies


def flag_anomalies(df: pd.DataFrame, features: List[str]) -> pd.Series:
    """
    Identify and flag anomalies in a DataFrame based on the Interquartile Range
    (IQR) method for specified features.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        features (List[str]): A list of column names in the DataFrame to check for
            anomalies.

    Returns:
        pd.Series: A Series of boolean values where True indicates an anomaly in
        any of the specified features.
    """
    anomaly_flags = pd.Series(False, index=df.index)

    for feature in features:
        first_quartile = df[feature].quantile(0.25)
        third_quartile = df[feature].quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * interquartile_range
        upper_bound = third_quartile + 1.5 * interquartile_range

        feature_anomalies = (df[feature] < lower_bound) | (
            df[feature] > upper_bound
        )
        anomaly_flags |= feature_anomalies

    return anomaly_flags


def calculate_cramers_v(x, y):
    """Calculates Cramer's V statistic for categorical-categorical association.

    Args:
        x: pandas Series
        y: pandas Series

    Returns:
        float: Cramer's V
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))


def handle_missing_values(data, threshold=0.3):
    """This function handles missing values in a given DataFrame by dropping
    columns and rows.

    Parameters:
    data (pandas.DataFrame): The input DataFrame with missing values.
    threshold (float, optional): The threshold for determining columns to drop. Default is 0.3.

    Returns:
    pandas.DataFrame: The cleaned DataFrame with missing values handled.
    """
    columns_to_drop = data.columns[data.isnull().mean() > threshold]
    data_cleaned = data.drop(columns=columns_to_drop)
    print(
        f"Columns dropped due to >{threshold*100}% missing values: {list(columns_to_drop)}"
    )

    data_cleaned = data_cleaned.dropna()
    print(
        f"Rows removed due to missing values: {len(data) - len(data_cleaned)}"
    )

    return data_cleaned


def simple_imputation(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform simple imputation on missing values in training and testing datasets.

    This function fills missing values in categorical columns with the mode
    (most frequent value) and in numerical columns with the median value. The
    imputation is done separately for the training and testing datasets.

    Args:
        train_data (pd.DataFrame): The training dataset containing missing values.
        test_data (pd.DataFrame): The testing dataset containing missing values.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The training dataset with imputed
        values and the testing dataset with imputed values.
    """
    train_data = train_data.copy()
    test_data = test_data.copy()

    cat_labels = train_data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns
    num_labels = train_data.select_dtypes(include=["int64", "float64"]).columns

    for na in cat_labels:
        train_data[na] = train_data[na].fillna(train_data[na].mode().iloc[0])
        if na in test_data.columns:
            test_data[na] = test_data[na].fillna(test_data[na].mode().iloc[0])

    for na in num_labels:
        train_data[na] = train_data[na].fillna(train_data[na].median())
        if na in test_data.columns:
            test_data[na] = test_data[na].fillna(test_data[na].median())

    return train_data, test_data


def engineer_spaceship_features(df: pd.DataFrame) -> pd.DataFrame:
    """Performs feature engineering on the input DataFrame and tags missing
    values.

    This function creates new features, extracts information from existing features,
    performs binning and interaction feature creation, and tags missing values.

    Args:
        df: The input DataFrame containing passenger information.

    Returns:
        The DataFrame with newly engineered features added and missing values tagged.

    The following features are created or modified:
        - TotalSpending: Sum of spending across various amenities
        - CabinDeck, CabinNumber, CabinSide: Extracted from Cabin information
        - GroupSize: Size of the group a passenger belongs to
        - AgeGroup: Binned age categories
        - HomePlanet_CryoSleep: Interaction feature between HomePlanet and CryoSleep
        - [FeatureName]_IsMissing: Binary flags indicating missing values for each feature
    """

    df["TotalSpending"] = (
        df["RoomService"]
        + df["FoodCourt"]
        + df["ShoppingMall"]
        + df["Spa"]
        + df["VRDeck"]
    )

    df["CabinDeck"] = df["Cabin"].str[0]
    df["CabinNumber"] = df["Cabin"].str.split("/").str[1].astype(float)
    df["CabinSide"] = df["Cabin"].str[-1]

    df["GroupId"] = df["PassengerId"].str.split("_").str[0]
    group_sizes = df.groupby("GroupId").size()
    df["GroupSize"] = df["GroupId"].map(group_sizes)
    df.drop("GroupId", axis=1, inplace=True)

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 18, 65, float("inf")],
        labels=["Child", "Adult", "Senior"],
    )

    df["HomePlanetCryoSleep"] = (
        df["HomePlanet"] + "_" + df["CryoSleep"].astype(str)
    )

    return df


def confidence_interval(
    data: List[float], confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculates the confidence interval for a given dataset.

    Args:
        data (List[float]): A list of numerical data points.
        confidence (float): The confidence level for the interval. Defaults to
            0.95.

    Returns:
        Tuple[float, float, float]: A tuple containing the mean, lower bound, and
        upper bound of the confidence interval.
    """
    array_data = np.array(data, dtype=float)
    sample_size = len(array_data)
    mean_value = np.mean(array_data)
    standard_error = stats.sem(array_data)
    margin_of_error = standard_error * stats.t.ppf(
        (1 + confidence) / 2.0, sample_size - 1
    )
    return (
        mean_value,
        mean_value - margin_of_error,
        mean_value + margin_of_error,
    )


def create_pipeline(preprocessor: Pipeline, model: Pipeline) -> Pipeline:
    """
    Create a machine learning pipeline with a preprocessor and a classifier.

    Parameters:
        preprocessor (sklearn.base.TransformerMixin): The preprocessing component
            of the pipeline.
        model (sklearn.base.BaseEstimator): The classifier component of the
            pipeline.

    Returns:
        sklearn.pipeline.Pipeline: A scikit-learn Pipeline object that
        sequentially applies the preprocessor and the classifier.
    """
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])
