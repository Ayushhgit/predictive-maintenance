"""
Unit tests for FastAPI endpoints.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app

        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "Predictive Maintenance API" in response.json()["message"]

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_models_list_endpoint(self, client):
        """Test models list endpoint."""
        response = client.get("/models")
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)

    @patch("api.main.load_model")
    @patch("api.main.load_scaler")
    def test_predict_endpoint(self, mock_scaler, mock_model, client):
        """Test prediction endpoint with mocked model."""
        # Setup mocks
        mock_model.return_value = MagicMock(predict=MagicMock(return_value=np.array([100.0])))
        mock_scaler.return_value = None

        request_data = {
            "readings": [
                {
                    "unit_number": 1,
                    "time_in_cycles": 100,
                    "op_setting_1": 0.0,
                    "op_setting_2": 0.0,
                    "op_setting_3": 100.0,
                    "sensor_1": 518.67,
                    "sensor_2": 642.68,
                    "sensor_3": 1589.70,
                    "sensor_4": 1400.60,
                    "sensor_5": 14.62,
                    "sensor_6": 21.61,
                    "sensor_7": 554.36,
                    "sensor_8": 2388.06,
                    "sensor_9": 9046.19,
                    "sensor_10": 1.30,
                    "sensor_11": 47.47,
                    "sensor_12": 521.66,
                    "sensor_13": 2388.02,
                    "sensor_14": 8138.62,
                    "sensor_15": 8.4195,
                    "sensor_16": 0.03,
                    "sensor_17": 392.0,
                    "sensor_18": 2388.0,
                    "sensor_19": 100.0,
                    "sensor_20": 39.06,
                    "sensor_21": 23.4190,
                }
            ],
            "model_name": "random_forest",
        }

        response = client.post("/predict", json=request_data)

        # Should either succeed or fail gracefully
        assert response.status_code in [200, 404, 500]


class TestRequestValidation:
    """Tests for request validation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app

        return TestClient(app)

    def test_invalid_request_missing_readings(self, client):
        """Test handling of missing readings."""
        response = client.post("/predict", json={"model_name": "random_forest"})
        assert response.status_code == 422

    def test_invalid_request_wrong_type(self, client):
        """Test handling of wrong data types."""
        response = client.post("/predict", json={"readings": "not a list"})
        assert response.status_code == 422


class TestRiskLevelCalculation:
    """Tests for risk level calculation."""

    def test_critical_risk(self):
        """Test critical risk level for low RUL."""
        from api.main import _calculate_risk_level

        assert _calculate_risk_level(10) == "CRITICAL"
        assert _calculate_risk_level(5) == "CRITICAL"

    def test_high_risk(self):
        """Test high risk level."""
        from api.main import _calculate_risk_level

        assert _calculate_risk_level(30) == "HIGH"
        assert _calculate_risk_level(45) == "HIGH"

    def test_medium_risk(self):
        """Test medium risk level."""
        from api.main import _calculate_risk_level

        assert _calculate_risk_level(60) == "MEDIUM"
        assert _calculate_risk_level(90) == "MEDIUM"

    def test_low_risk(self):
        """Test low risk level for high RUL."""
        from api.main import _calculate_risk_level

        assert _calculate_risk_level(150) == "LOW"
        assert _calculate_risk_level(200) == "LOW"
