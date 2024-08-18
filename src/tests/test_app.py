"""Unit tests for the Spaceship Titanic Prediction API."""

import json
import unittest
from unittest.mock import patch

import pandas as pd

from app import (
    app,
    SpaceshipPassenger,
    preprocess_data,
    detect_anomalies_iqr,
    engineer_spaceship_features,
)


class TestSpaceshipTitanicAPI(unittest.TestCase):
    """Test cases for the Spaceship Titanic Prediction API."""

    def setUp(self):
        """Set up test client and other test variables."""
        self.app = app.test_client()
        self.app.testing = True

    def test_home_route(self):
        """Test the home route."""
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Spaceship Titanic Predictor", response.data)

    @patch("app.predict")
    def test_predict_route_valid_input(self, mock_predict):
        """Test the predict route with valid input."""
        mock_predict.return_value = {
            "transported": True,
            "transported_probability": 0.75,
        }
        data = {
            "home_planet": "Earth",
            "cryo_sleep": False,
            "cabin": "B/0/P",
            "destination": "TRAPPIST-1e",
            "age": 30,
            "vip": False,
            "room_service": 100,
            "food_court": 50,
            "shopping_mall": 20,
            "spa": 30,
            "vr_deck": 40,
        }
        response = self.app.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            json.loads(response.data),
            {"transported": True, "transported_probability": 0.75},
        )

    def test_predict_route_invalid_input(self):
        """Test the predict route with invalid input."""
        data = {"invalid_field": "value"}
        response = self.app.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", json.loads(response.data))

    @patch("app.MODEL")
    def test_model_info_route(self, mock_model):
        """Test the model-info route."""
        mock_model.model_json = {"output": {"names": ["feature1", "feature2"]}}
        response = self.app.get("/model-info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            json.loads(response.data),
            {"expected_columns": ["feature1", "feature2"]},
        )

    def test_preprocess_data(self):
        """Test the preprocess_data function."""
        passenger = SpaceshipPassenger(
            home_planet="Earth",
            cryo_sleep=False,
            cabin="B/0/P",
            destination="TRAPPIST-1e",
            age=30,
            vip=False,
            room_service=100,
            food_court=50,
            shopping_mall=20,
            spa=30,
            vr_deck=40,
        )
        result = preprocess_data(passenger)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("TotalSpending", result.columns)
        self.assertIn("CabinDeck", result.columns)

    def test_detect_anomalies_iqr(self):
        """Test the detect_anomalies_iqr function."""
        df = pd.DataFrame({"A": [1, 2, 3, 4, 100], "B": [10, 20, 30, 40, 50]})
        result = detect_anomalies_iqr(df, ["A", "B"])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)  # One anomaly in column A

    def test_engineer_spaceship_features(self):
        """Test the engineer_spaceship_features function."""
        df = pd.DataFrame(
            {
                "HomePlanet": ["Earth"],
                "CryoSleep": [False],
                "RoomService": [100],
                "FoodCourt": [50],
                "ShoppingMall": [20],
                "Spa": [30],
                "VRDeck": [40],
                "Cabin": ["B/0/P"],
                "Age": [30],
            }
        )
        result = engineer_spaceship_features(df)
        self.assertIn("TotalSpending", result.columns)
        self.assertIn("CabinDeck", result.columns)
        self.assertIn("AgeGroup", result.columns)
        self.assertIn("HomePlanetCryoSleep", result.columns)


if __name__ == "__main__":
    unittest.main()
