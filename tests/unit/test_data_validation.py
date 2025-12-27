"""
Unit tests for data validation component.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestDataValidation:
    """Tests for DataValidation class."""

    def test_sample_data_shape(self, sample_sensor_data):
        """Test that sample data has expected shape."""
        assert sample_sensor_data.shape[0] == 100
        assert "unit_number" in sample_sensor_data.columns
        assert "time_in_cycles" in sample_sensor_data.columns

    def test_sample_data_sensors(self, sample_sensor_data):
        """Test that sample data has all sensor columns."""
        sensor_cols = [col for col in sample_sensor_data.columns if col.startswith("sensor_")]
        assert len(sensor_cols) == 21

    def test_sample_data_no_nulls(self, sample_sensor_data):
        """Test that sample data has no null values."""
        assert sample_sensor_data.isnull().sum().sum() == 0

    def test_sample_data_types(self, sample_sensor_data):
        """Test that sample data has correct types."""
        assert sample_sensor_data["unit_number"].dtype in [np.int64, np.int32]
        assert sample_sensor_data["sensor_1"].dtype == np.float64


class TestDataPreprocessing:
    """Tests for data preprocessing functions."""

    def test_feature_scaling(self, sample_features):
        """Test feature scaling."""
        X, _ = sample_features
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Check mean is approximately 0
        assert np.abs(X_scaled.mean()) < 0.1

        # Check std is approximately 1
        assert np.abs(X_scaled.std() - 1) < 0.1

    def test_train_test_split(self, sample_features):
        """Test train-test split."""
        from sklearn.model_selection import train_test_split

        X, y = sample_features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20


class TestDataQuality:
    """Tests for data quality checks."""

    def test_outlier_detection(self, sample_sensor_data):
        """Test outlier detection."""
        sensor_col = "sensor_1"
        data = sample_sensor_data[sensor_col]

        # Calculate z-scores
        z_scores = np.abs((data - data.mean()) / data.std())

        # Count outliers (z-score > 3)
        outliers = (z_scores > 3).sum()

        # With random normal data, expect few outliers
        assert outliers <= len(data) * 0.01  # Less than 1%

    def test_missing_value_handling(self):
        """Test missing value handling."""
        # Create data with missing values
        data = pd.DataFrame({"A": [1, 2, np.nan, 4, 5], "B": [np.nan, 2, 3, 4, 5]})

        # Forward fill
        filled = data.ffill().bfill()

        assert filled.isnull().sum().sum() == 0

    def test_duplicate_detection(self, sample_sensor_data):
        """Test duplicate row detection."""
        # Create data with duplicates
        data_with_dups = pd.concat([sample_sensor_data, sample_sensor_data.head(5)])

        duplicates = data_with_dups.duplicated().sum()
        assert duplicates == 5

        # Remove duplicates
        cleaned = data_with_dups.drop_duplicates()
        assert len(cleaned) == len(sample_sensor_data)
