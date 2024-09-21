"""Unit tests for the basic_visualizations_utils module."""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from spaceship_titanic.basic_visualizations_utils import (
    plot_combined_histograms,
    plot_combined_bar_charts,
    plot_combined_boxplots,
    plot_correlation_matrix,
    plot_feature_importances,
    plot_distribution_comparison,
    plot_categorical_features_by_target,
    plot_numeric_distributions,
    plot_single_bar_chart,
)


class TestBasicVisualizationsUtils(unittest.TestCase):
    """Test cases for basic_visualizations_utils module."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
                "C": ["x", "y", "z", "x", "y"],
            }
        )
        self.features = ["A", "B"]
        self.categorical_features = ["C"]

    @patch("spaceship_titanic.basic_visualizations_utils.make_subplots")
    @patch("spaceship_titanic.basic_visualizations_utils.go.Histogram")
    def test_plot_combined_histograms(self, mock_histogram, mock_make_subplots):
        """Test plot_combined_histograms function."""
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig

        plot_combined_histograms(self.df, self.features)

        self.assertEqual(mock_histogram.call_count, len(self.features))
        mock_fig.show.assert_called_once()

    @patch("spaceship_titanic.basic_visualizations_utils.make_subplots")
    @patch("spaceship_titanic.basic_visualizations_utils.go.Bar")
    def test_plot_combined_bar_charts(self, mock_bar, mock_make_subplots):
        """Test plot_combined_bar_charts function."""
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig

        plot_combined_bar_charts(self.df, self.categorical_features)

        mock_bar.assert_called_once()
        mock_fig.show.assert_called_once()

    @patch("spaceship_titanic.basic_visualizations_utils.make_subplots")
    @patch("spaceship_titanic.basic_visualizations_utils.go.Box")
    def test_plot_combined_boxplots(self, mock_box, mock_make_subplots):
        """Test plot_combined_boxplots function."""
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig

        plot_combined_boxplots(self.df, self.features)

        self.assertEqual(mock_box.call_count, len(self.features))
        mock_fig.show.assert_called_once()

    @patch("spaceship_titanic.basic_visualizations_utils.px.imshow")
    def test_plot_correlation_matrix(self, mock_imshow):
        """Test plot_correlation_matrix function."""
        mock_fig = MagicMock()
        mock_imshow.return_value = mock_fig

        plot_correlation_matrix(self.df, self.features)

        mock_imshow.assert_called_once()
        mock_fig.show.assert_called_once()

    @patch("spaceship_titanic.basic_visualizations_utils.go.Figure")
    def test_plot_feature_importances(self, mock_figure):
        """Test plot_feature_importances function."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig

        feature_importances = {
            "Model1": {"A": 0.5, "B": 0.3},
            "Model2": {"A": 0.4, "B": 0.6},
        }

        plot_feature_importances(feature_importances)

        mock_figure.assert_called_once()
        mock_fig.show.assert_called_once()

    @patch("spaceship_titanic.basic_visualizations_utils.make_subplots")
    def test_plot_distribution_comparison(self, mock_make_subplots):
        """Test plot_distribution_comparison function."""
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig

        df_before = self.df.copy()
        df_after = self.df.copy()
        df_after["A"] = df_after["A"] + 1

        plot_distribution_comparison(df_before, df_after, self.features)

        mock_make_subplots.assert_called_once()
        mock_fig.show.assert_called_once()

    @patch("spaceship_titanic.basic_visualizations_utils.make_subplots")
    def test_plot_categorical_features_by_target(self, mock_make_subplots):
        """Test plot_categorical_features_by_target function."""
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig

        self.df["target"] = [0, 1, 0, 1, 0]

        plot_categorical_features_by_target(
            self.df, self.categorical_features, "target"
        )

        mock_make_subplots.assert_called_once()
        mock_fig.show.assert_called_once()

    @patch("spaceship_titanic.basic_visualizations_utils.make_subplots")
    def test_plot_numeric_distributions(self, mock_make_subplots):
        """Test plot_numeric_distributions function."""
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig

        self.df["target"] = [0, 1, 0, 1, 0]

        plot_numeric_distributions(self.df, self.features, "target")

        mock_make_subplots.assert_called_once()
        mock_fig.show.assert_called_once()

    @patch("spaceship_titanic.basic_visualizations_utils.go.Figure")
    def test_plot_single_bar_chart(self, mock_figure):
        """Test plot_single_bar_chart function."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig

        plot_single_bar_chart(self.df, "C")

        mock_figure.assert_called_once()
        mock_fig.show.assert_called_once()


if __name__ == "__main__":
    unittest.main()
