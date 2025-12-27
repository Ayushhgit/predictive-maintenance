"""
Pytest configuration and fixtures for testing.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_sensor_data():
    """Generate sample sensor data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "unit_number": np.repeat(range(1, 6), 20),
        "time_in_cycles": np.tile(range(1, 21), 5),
        "op_setting_1": np.random.uniform(-0.1, 0.1, n_samples),
        "op_setting_2": np.random.uniform(-0.1, 0.1, n_samples),
        "op_setting_3": np.random.uniform(99, 101, n_samples),
    }

    # Add 21 sensor readings
    for i in range(1, 22):
        data[f"sensor_{i}"] = np.random.uniform(0, 100, n_samples)

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def sample_features():
    """Generate sample features for model testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features), columns=[f"feature_{i}" for i in range(n_features)]
    )
    y = pd.Series(np.random.uniform(10, 200, n_samples), name="RUL")

    return X, y


@pytest.fixture(scope="session")
def sample_predictions():
    """Generate sample predictions for evaluation testing."""
    np.random.seed(42)
    n_samples = 100

    y_true = np.random.uniform(10, 200, n_samples)
    y_pred = y_true + np.random.normal(0, 10, n_samples)

    return y_true, y_pred


@pytest.fixture
def mock_model():
    """Create a mock ML model."""
    model = MagicMock()
    model.predict.return_value = np.array([100, 150, 200, 50, 75])
    model.fit.return_value = model
    return model


@pytest.fixture
def temp_config(tmp_path):
    """Create temporary config file."""
    config_content = """
data:
  raw_data_path: "data/raw/"
  transformed_data_path: "data/transformed/"
  validated_data_path: "data/validated/"

model:
  target_column: "RUL"
  test_size: 0.2
  lstm:
    sequence_length: 10
    units: 64
    dropout: 0.2
    epochs: 5
    batch_size: 32

artifacts:
  model_dir: "artifacts/models/"
  logs_dir: "artifacts/logs/"
  reports_dir: "artifacts/reports/"

features:
  sensor_columns:
    - "sensor_1"
    - "sensor_2"
  window_sizes: [5, 10]
  lag_features: [1, 3]
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return str(config_path)


@pytest.fixture
def temp_data_dir(tmp_path, sample_sensor_data):
    """Create temporary data directory with sample data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Save sample data
    sample_sensor_data.to_csv(data_dir / "train.csv", index=False)
    sample_sensor_data.to_csv(data_dir / "test.csv", index=False)

    return str(data_dir)


@pytest.fixture
def mock_config():
    """Return mock configuration dictionary."""
    return {
        "data": {
            "raw_data_path": "data/raw/",
            "transformed_data_path": "data/transformed/",
            "validated_data_path": "data/validated/",
        },
        "model": {
            "target_column": "RUL",
            "test_size": 0.2,
            "lstm": {
                "sequence_length": 10,
                "units": 64,
                "dropout": 0.2,
                "epochs": 5,
                "batch_size": 32,
            },
        },
        "artifacts": {
            "model_dir": "artifacts/models/",
            "logs_dir": "artifacts/logs/",
            "reports_dir": "artifacts/reports/",
        },
        "features": {
            "sensor_columns": ["sensor_1", "sensor_2"],
            "window_sizes": [5, 10],
            "lag_features": [1, 3],
        },
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
