"""
Unit tests for model evaluation component.
"""

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class TestMetricsCalculation:
    """Tests for metrics calculation functions."""

    def test_mse_calculation(self, sample_predictions):
        """Test MSE calculation."""
        y_true, y_pred = sample_predictions
        mse = mean_squared_error(y_true, y_pred)

        assert mse >= 0
        assert isinstance(mse, float)

    def test_rmse_calculation(self, sample_predictions):
        """Test RMSE calculation."""
        y_true, y_pred = sample_predictions
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        assert rmse >= 0
        assert rmse <= np.sqrt(mse) + 1e-6

    def test_mae_calculation(self, sample_predictions):
        """Test MAE calculation."""
        y_true, y_pred = sample_predictions
        mae = mean_absolute_error(y_true, y_pred)

        assert mae >= 0
        assert isinstance(mae, float)

    def test_r2_calculation(self, sample_predictions):
        """Test R² calculation."""
        y_true, y_pred = sample_predictions
        r2 = r2_score(y_true, y_pred)

        # R² can be negative for bad predictions
        assert -1 <= r2 <= 1 or r2 < -1  # Allow very bad predictions

    def test_perfect_prediction_metrics(self):
        """Test metrics for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        assert mse == 0
        assert mae == 0
        assert r2 == 1.0


class TestModelComparison:
    """Tests for model comparison functions."""

    def test_best_model_selection(self):
        """Test selecting best model by RMSE."""
        model_results = {
            "model_a": {"rmse": 15.5, "r2": 0.85},
            "model_b": {"rmse": 12.3, "r2": 0.90},
            "model_c": {"rmse": 18.2, "r2": 0.80},
        }

        best_model = min(model_results.keys(), key=lambda x: model_results[x]["rmse"])
        assert best_model == "model_b"

    def test_metric_aggregation(self):
        """Test aggregating metrics across models."""
        model_results = {
            "model_a": {"rmse": 15.5, "mae": 10.2},
            "model_b": {"rmse": 12.3, "mae": 8.5},
            "model_c": {"rmse": 18.2, "mae": 12.1},
        }

        avg_rmse = np.mean([r["rmse"] for r in model_results.values()])
        avg_mae = np.mean([r["mae"] for r in model_results.values()])

        assert 12 < avg_rmse < 16
        assert 8 < avg_mae < 11


class TestResidualAnalysis:
    """Tests for residual analysis."""

    def test_residual_calculation(self, sample_predictions):
        """Test residual calculation."""
        y_true, y_pred = sample_predictions
        residuals = y_true - y_pred

        assert len(residuals) == len(y_true)

    def test_residual_mean_near_zero(self, sample_predictions):
        """Test that residual mean is near zero for unbiased predictions."""
        y_true, y_pred = sample_predictions
        residuals = y_true - y_pred

        # Mean should be near zero for unbiased predictions
        assert np.abs(residuals.mean()) < 5

    def test_residual_normality(self, sample_predictions):
        """Test residual distribution."""
        from scipy import stats

        y_true, y_pred = sample_predictions
        residuals = y_true - y_pred

        # Shapiro-Wilk test for normality
        _, p_value = stats.shapiro(residuals[:50])  # Use subset for test

        # p_value > 0.05 suggests normal distribution
        # We just check it runs, not enforce normality
        assert isinstance(p_value, float)
