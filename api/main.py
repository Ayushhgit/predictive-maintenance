"""
FastAPI Application for Predictive Maintenance Model Serving
Provides REST API endpoints for RUL predictions and model management.
"""

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from src.constants import CONFIG_FILE_PATH, MODEL_DIR
from src.utils.logger import get_logger
from src.utils.model_utils import load_yaml

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="REST API for Remaining Useful Life (RUL) prediction of industrial equipment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Health", "description": "Health check endpoints"},
        {"name": "Predictions", "description": "Model prediction endpoints"},
        {"name": "Models", "description": "Model management endpoints"},
    ],
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
model_cache: Dict[str, Any] = {}
scaler_cache: Any = None


# Pydantic models for request/response
class SensorReading(BaseModel):
    """Single sensor reading with all sensor values."""

    unit_number: int = Field(..., description="Equipment unit identifier")
    time_in_cycles: int = Field(..., description="Operating time in cycles")
    op_setting_1: float = Field(..., description="Operational setting 1")
    op_setting_2: float = Field(..., description="Operational setting 2")
    op_setting_3: float = Field(..., description="Operational setting 3")
    sensor_1: float = Field(..., description="Sensor 1 reading")
    sensor_2: float = Field(..., description="Sensor 2 reading")
    sensor_3: float = Field(..., description="Sensor 3 reading")
    sensor_4: float = Field(..., description="Sensor 4 reading")
    sensor_5: float = Field(..., description="Sensor 5 reading")
    sensor_6: float = Field(..., description="Sensor 6 reading")
    sensor_7: float = Field(..., description="Sensor 7 reading")
    sensor_8: float = Field(..., description="Sensor 8 reading")
    sensor_9: float = Field(..., description="Sensor 9 reading")
    sensor_10: float = Field(..., description="Sensor 10 reading")
    sensor_11: float = Field(..., description="Sensor 11 reading")
    sensor_12: float = Field(..., description="Sensor 12 reading")
    sensor_13: float = Field(..., description="Sensor 13 reading")
    sensor_14: float = Field(..., description="Sensor 14 reading")
    sensor_15: float = Field(..., description="Sensor 15 reading")
    sensor_16: float = Field(..., description="Sensor 16 reading")
    sensor_17: float = Field(..., description="Sensor 17 reading")
    sensor_18: float = Field(..., description="Sensor 18 reading")
    sensor_19: float = Field(..., description="Sensor 19 reading")
    sensor_20: float = Field(..., description="Sensor 20 reading")
    sensor_21: float = Field(..., description="Sensor 21 reading")


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""

    readings: List[SensorReading] = Field(..., description="List of sensor readings")
    model_name: str = Field(default="random_forest", description="Model to use for prediction")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response body for prediction endpoint."""

    predictions: List[Dict[str, Any]] = Field(..., description="Prediction results")
    model_used: str = Field(..., description="Model name used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    api_version: str = Field(default="1.0.0", description="API version")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    models_loaded: List[str]
    version: str


class ModelInfo(BaseModel):
    """Model information response."""

    name: str
    loaded: bool
    type: str


def load_model(model_name: str) -> Any:
    """Load model from cache or disk."""
    if model_name in model_cache:
        return model_cache[model_name]

    model_path = Path(MODEL_DIR) / f"{model_name}_model.pkl"
    if not model_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Model '{model_name}' not found at {model_path}"
        )

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        model_cache[model_name] = model
        logger.info(f"Loaded model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


def load_scaler() -> Any:
    """Load the feature scaler."""
    global scaler_cache
    if scaler_cache is not None:
        return scaler_cache

    scaler_path = Path(MODEL_DIR) / "scaler.pkl"
    if not scaler_path.exists():
        logger.warning("Scaler not found, proceeding without scaling")
        return None

    try:
        with open(scaler_path, "rb") as f:
            scaler_cache = pickle.load(f)
        logger.info("Loaded feature scaler")
        return scaler_cache
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        return None


def preprocess_input(readings: List[SensorReading], scaler: Any = None) -> np.ndarray:
    """Preprocess input data for prediction."""
    # Convert to DataFrame
    data = [reading.model_dump() for reading in readings]
    df = pd.DataFrame(data)

    # Select only feature columns (exclude unit_number and time_in_cycles for prediction)
    feature_cols = [col for col in df.columns if col not in ["unit_number", "time_in_cycles"]]
    X = df[feature_cols].values

    # Apply scaling if scaler is available
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            logger.warning(f"Scaling failed, using raw features: {e}")

    return X


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("Starting Predictive Maintenance API...")

    # Pre-load default model
    try:
        load_model("random_forest")
        load_scaler()
        logger.info("API started successfully")
    except Exception as e:
        logger.warning(f"Could not pre-load models: {e}")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {"message": "Predictive Maintenance API", "docs": "/docs", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=list(model_cache.keys()),
        version="1.0.0",
    )


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """List available models."""
    available_models = [
        "random_forest",
        "gradient_boosting",
        "linear_regression",
        "ridge",
        "lasso",
        "svr",
    ]

    model_info = []
    for model_name in available_models:
        model_path = Path(MODEL_DIR) / f"{model_name}_model.pkl"
        model_info.append(
            ModelInfo(name=model_name, loaded=model_name in model_cache, type="sklearn")
        )

    return model_info


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Make RUL predictions for sensor readings.

    - **readings**: List of sensor readings from equipment
    - **model_name**: Name of the model to use (default: random_forest)
    """
    try:
        # Load model
        model = load_model(request.model_name)
        scaler = load_scaler()

        # Preprocess input
        X = preprocess_input(request.readings, scaler)

        # Make predictions
        predictions = model.predict(X)

        # Format response
        results = []
        for i, (reading, pred) in enumerate(zip(request.readings, predictions)):
            results.append(
                {
                    "unit_number": reading.unit_number,
                    "time_in_cycles": reading.time_in_cycles,
                    "predicted_rul": float(max(0, pred)),  # RUL can't be negative
                    "risk_level": _calculate_risk_level(pred),
                }
            )

        return PredictionResponse(
            predictions=results,
            model_used=request.model_name,
            timestamp=datetime.now().isoformat(),
            api_version="1.0.0",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Predictions"])
async def batch_predict(file_path: str, model_name: str = "random_forest"):
    """
    Make batch predictions from a CSV file.

    - **file_path**: Path to CSV file with sensor readings
    - **model_name**: Name of the model to use
    """
    try:
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        model = load_model(model_name)
        scaler = load_scaler()

        # Select feature columns
        feature_cols = [
            col for col in df.columns if col.startswith("sensor_") or col.startswith("op_setting_")
        ]
        X = df[feature_cols].values

        if scaler:
            X = scaler.transform(X)

        predictions = model.predict(X)

        results = {
            "total_samples": len(predictions),
            "predictions": predictions.tolist(),
            "statistics": {
                "mean_rul": float(np.mean(predictions)),
                "min_rul": float(np.min(predictions)),
                "max_rul": float(np.max(predictions)),
                "std_rul": float(np.std(predictions)),
            },
            "model_used": model_name,
            "timestamp": datetime.now().isoformat(),
        }

        return JSONResponse(content=results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/reload", tags=["Models"])
async def reload_models():
    """Reload all models from disk."""
    global model_cache, scaler_cache
    model_cache = {}
    scaler_cache = None

    try:
        load_model("random_forest")
        load_scaler()
        return {
            "message": "Models reloaded successfully",
            "models_loaded": list(model_cache.keys()),
        }
    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _calculate_risk_level(rul: float) -> str:
    """Calculate risk level based on RUL."""
    if rul < 20:
        return "CRITICAL"
    elif rul < 50:
        return "HIGH"
    elif rul < 100:
        return "MEDIUM"
    else:
        return "LOW"


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
