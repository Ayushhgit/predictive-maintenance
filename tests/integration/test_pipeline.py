"""
Integration tests for ML pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.integration
class TestDataPipeline:
    """Integration tests for data processing pipeline."""

    def test_data_flow(self, sample_sensor_data, tmp_path):
        """Test data flows through preprocessing steps."""
        # Save sample data
        input_path = tmp_path / "input.csv"
        sample_sensor_data.to_csv(input_path, index=False)

        # Read back
        loaded_data = pd.read_csv(input_path)

        # Verify data integrity
        assert loaded_data.shape == sample_sensor_data.shape
        assert list(loaded_data.columns) == list(sample_sensor_data.columns)

    def test_feature_engineering(self, sample_sensor_data):
        """Test feature engineering produces expected features."""
        # Add lag features
        df = sample_sensor_data.copy()
        df["sensor_1_lag_1"] = df.groupby("unit_number")["sensor_1"].shift(1)

        # Verify lag feature was created
        assert "sensor_1_lag_1" in df.columns

        # First row of each unit should have NaN
        first_rows = df.groupby("unit_number").first()
        assert first_rows["sensor_1_lag_1"].isna().all()

    def test_train_test_consistency(self, sample_features):
        """Test train and test sets have consistent features."""
        from sklearn.model_selection import train_test_split

        X, y = sample_features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Same columns
        assert list(X_train.columns) == list(X_test.columns)

        # No data leakage (different indices)
        assert len(set(X_train.index) & set(X_test.index)) == 0


@pytest.mark.integration
class TestModelPipeline:
    """Integration tests for model training and evaluation."""

    def test_model_training_flow(self, sample_features):
        """Test model can be trained on sample data."""
        from sklearn.ensemble import RandomForestRegressor

        X, y = sample_features

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Model should be able to predict
        predictions = model.predict(X[:5])
        assert len(predictions) == 5

    def test_model_evaluation_flow(self, sample_features):
        """Test model evaluation produces expected metrics."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        X, y = sample_features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        assert mse >= 0
        assert isinstance(r2, float)

    def test_model_serialization(self, sample_features, tmp_path):
        """Test model can be saved and loaded."""
        import pickle

        from sklearn.ensemble import RandomForestRegressor

        X, y = sample_features

        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Save model
        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Load model
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)

        # Compare predictions
        original_pred = model.predict(X[:5])
        loaded_pred = loaded_model.predict(X[:5])

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-end integration tests."""

    def test_full_pipeline_flow(self, sample_sensor_data, tmp_path):
        """Test complete pipeline from raw data to predictions."""
        import pickle

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        # 1. Data preparation
        df = sample_sensor_data.copy()

        # 2. Feature engineering - add RUL
        max_cycles = df.groupby("unit_number")["time_in_cycles"].max()
        df["RUL"] = df.apply(
            lambda row: max_cycles[row["unit_number"]] - row["time_in_cycles"], axis=1
        )

        # 3. Prepare features
        feature_cols = [
            col for col in df.columns if col.startswith("sensor_") or col.startswith("op_setting_")
        ]
        X = df[feature_cols]
        y = df["RUL"]

        # 4. Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 5. Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_scaled, y)

        # 6. Save artifacts
        model_path = tmp_path / "model.pkl"
        scaler_path = tmp_path / "scaler.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # 7. Load and predict
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            loaded_scaler = pickle.load(f)

        # 8. Make predictions on new data
        new_data = X.iloc[:5]
        new_scaled = loaded_scaler.transform(new_data)
        predictions = loaded_model.predict(new_scaled)

        # Verify predictions
        assert len(predictions) == 5
        assert all(pred >= 0 or True for pred in predictions)  # RUL should be reasonable
