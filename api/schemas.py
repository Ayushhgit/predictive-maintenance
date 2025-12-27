"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SensorReadingBase(BaseModel):
    """Base sensor reading schema."""

    unit_number: int
    time_in_cycles: int
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float


class SensorReadingFull(SensorReadingBase):
    """Full sensor reading with all 21 sensors."""

    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class PredictionResult(BaseModel):
    """Single prediction result."""

    unit_number: int
    time_in_cycles: int
    predicted_rul: float
    risk_level: str
    confidence: Optional[float] = None


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""

    readings: List[SensorReadingFull]
    model_name: str = "random_forest"


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    predictions: List[PredictionResult]
    model_used: str
    total_samples: int
    processing_time_ms: float
    timestamp: str


class ModelMetrics(BaseModel):
    """Model performance metrics."""

    mse: float
    rmse: float
    mae: float
    r2: float


class ModelDetails(BaseModel):
    """Detailed model information."""

    name: str
    type: str
    loaded: bool
    metrics: Optional[ModelMetrics] = None
    last_trained: Optional[str] = None


class HealthStatus(BaseModel):
    """API health status."""

    status: str
    uptime_seconds: float
    models_loaded: int
    last_prediction: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    detail: str
    timestamp: str
