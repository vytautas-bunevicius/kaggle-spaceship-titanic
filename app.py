#cspell:disable
"""Spaceship Titanic Prediction API."""

import logging
import traceback
from typing import Dict, Any

from flask import Flask, request, jsonify, render_template
from pydantic import BaseModel, Field
import h2o
import pandas as pd
import numpy as np

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Initialize MODEL at the module level
MODEL = None

class SpaceshipPassenger(BaseModel):
    """Represents a passenger on the Spaceship Titanic."""

    home_planet: str = Field(..., description="Passenger's home planet")
    cryo_sleep: bool = Field(..., description="Whether the passenger was in cryo sleep")
    cabin: str = Field(..., description="Passenger's cabin")
    destination: str = Field(..., description="Passenger's destination")
    age: int = Field(..., description="Passenger's age")
    vip: bool = Field(..., description="Whether the passenger is a VIP")
    room_service: int = Field(0, description="Amount spent on room service")
    food_court: int = Field(0, description="Amount spent at food court")
    shopping_mall: int = Field(0, description="Amount spent at shopping mall")
    spa: int = Field(0, description="Amount spent at spa")
    vr_deck: int = Field(0, description="Amount spent on VR deck")
    name: str = Field(None, description="Passenger's name")

def detect_anomalies_iqr(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Detect anomalies using the Interquartile Range method."""
    anomalies_list = []

    for feature in features:
        if feature not in df.columns:
            logging.warning("Feature '%s' not found in DataFrame.", feature)
            continue

        if not np.issubdtype(df[feature].dtype, np.number):
            logging.warning("Feature '%s' is not numerical and will be skipped.", feature)
            continue

        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        feature_anomalies = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]

        if not feature_anomalies.empty:
            logging.info("Anomalies detected in feature '%s': %s", feature, feature_anomalies)
        else:
            logging.info("No anomalies detected in feature '%s'.", feature)

        anomalies_list.append(feature_anomalies)

    if anomalies_list:
        anomalies = pd.concat(anomalies_list).drop_duplicates().reset_index(drop=True)
        return anomalies[features]
    return pd.DataFrame(columns=features)

def engineer_spaceship_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for the Spaceship Titanic dataset."""
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

    df["GroupSize"] = 1  # Since we're dealing with a single passenger, set GroupSize to 1

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 18, 65, float("inf")],
        labels=["Child", "Adult", "Senior"],
    )

    df["HomePlanetCryoSleep"] = df["HomePlanet"] + "_" + df["CryoSleep"].astype(str)

    return df

def preprocess_data(passenger: SpaceshipPassenger) -> pd.DataFrame:
    """Preprocess passenger data for prediction."""
    data = pd.DataFrame([passenger.dict()])

    # Rename columns to match feature names expected in the data engineering
    column_mapping = {
        "home_planet": "HomePlanet",
        "cryo_sleep": "CryoSleep",
        "destination": "Destination",
        "age": "Age",
        "vip": "VIP",
        "room_service": "RoomService",
        "food_court": "FoodCourt",
        "shopping_mall": "ShoppingMall",
        "spa": "Spa",
        "vr_deck": "VRDeck",
        "cabin": "Cabin",
        "name": "Name"
    }
    data.rename(columns=column_mapping, inplace=True)

    # Perform feature engineering
    data = engineer_spaceship_features(data)

    # Detect anomalies
    numerical_features = [
        "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
        "TotalSpending", "CabinNumber"
    ]
    anomalies = detect_anomalies_iqr(data, numerical_features)
    data["IsAnomaly"] = data.index.isin(anomalies.index).astype(int)

    return data

def predict(passenger: SpaceshipPassenger) -> Dict[str, Any]:
    """Make a prediction for a passenger."""
    preprocessed_data = preprocess_data(passenger)
    logging.debug("Preprocessed data: %s", preprocessed_data.to_dict(orient='records'))

    # Convert to H2OFrame
    h2o_frame = h2o.H2OFrame(preprocessed_data)

    # Ensure all columns from the training data are present
    model_columns = [
        'PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP',
        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'IsAnomaly',
        'TotalSpending', 'CabinDeck', 'CabinNumber', 'CabinSide', 'GroupSize', 'AgeGroup',
        'HomePlanetCryoSleep'
    ]
    for col in model_columns:
        if col not in h2o_frame.columns:
            h2o_frame[col] = None

    # Convert categorical columns to enum type
    categorical_columns = [
        'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'IsAnomaly',
        'CabinDeck', 'CabinSide', 'AgeGroup', 'HomePlanetCryoSleep'
    ]
    for col in categorical_columns:
        h2o_frame[col] = h2o_frame[col].asfactor()

    predictions = MODEL.predict(h2o_frame)

    logging.debug("Prediction columns: %s", predictions.columns)

    if "predict" in predictions.columns and "True" in predictions.columns:
        transported = bool(predictions["predict"][0, 0])
        probability = float(predictions["True"][0, 0])  # Probability of being transported
    else:
        raise ValueError("Unexpected prediction structure")

    return {
        "transported": transported,
        "transported_probability": probability
    }

@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def make_prediction():
    """Make a prediction based on input data."""
    try:
        logging.debug("Received data: %s", request.json)
        passenger = SpaceshipPassenger(**request.json)
        result = predict(passenger)
        logging.debug("Prediction result: %s", result)
        return jsonify(result)
    except (ValueError, KeyError, TypeError) as error:
        logging.error("Error during prediction: %s", str(error), exc_info=True)
        return jsonify({"error": str(error), "details": traceback.format_exc()}), 400

@app.route("/model-info", methods=["GET"])
def model_info():
    """Get information about the model."""
    try:
        model_columns = MODEL.model_json['output']['names']
        return jsonify({"expected_columns": model_columns})
    except (AttributeError, KeyError) as error:
        logging.error("Error fetching model info: %s", str(error), exc_info=True)
        return jsonify({"error": str(error), "details": traceback.format_exc()}), 400

if __name__ == "__main__":
    h2o.init()
    MODEL = h2o.load_model("/app/models/StackedEnsemble_Best1000_1_AutoML_1_20240811_214618")
    app.run(host="0.0.0.0", port=8080, debug=True)
