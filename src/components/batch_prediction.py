"""
Batch Prediction Component
Handles batch predictions for large datasets and scheduled inference.
"""

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from src.constants import CONFIG_FILE_PATH, LSTM_MODEL_NAME, MODEL_DIR, PREDICTIONS_DIR
from src.utils.logger import get_logger
from src.utils.model_utils import create_sequences_for_regression, load_csv, load_pickle, load_yaml

logger = get_logger(__name__)


@dataclass
class BatchPredictionConfig:
    """Configuration for batch prediction."""

    model_dir: str
    predictions_dir: str
    sequence_length: int
    default_model: str = "random_forest"
    batch_size: int = 1000
    save_predictions: bool = True


class BatchPredictor:
    """Handles batch predictions for production inference."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path or CONFIG_FILE_PATH)
        self.models = {}
        self.scaler = None
        self._load_artifacts()

    def _load_config(self, config_path: str) -> BatchPredictionConfig:
        """Load configuration from YAML file."""
        try:
            config_dict = load_yaml(config_path)
            return BatchPredictionConfig(
                model_dir=config_dict["artifacts"]["model_dir"],
                predictions_dir=PREDICTIONS_DIR,
                sequence_length=config_dict["model"]["lstm"]["sequence_length"],
            )
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise e

    def _load_artifacts(self) -> None:
        """Load models and scaler."""
        try:
            # Load scaler
            scaler_path = Path(self.config.model_dir) / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = load_pickle(str(scaler_path))
                logger.info("Scaler loaded successfully")

            # Load available models
            model_dir = Path(self.config.model_dir)
            for model_file in model_dir.glob("*_model.pkl"):
                model_name = model_file.stem.replace("_model", "")
                self.models[model_name] = load_pickle(str(model_file))
                logger.info(f"Loaded model: {model_name}")

        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise e

    def _load_lstm_model(self) -> Optional[tf.keras.Model]:
        """Load LSTM model if available."""
        lstm_path = Path(self.config.model_dir) / LSTM_MODEL_NAME
        if lstm_path.exists():
            return tf.keras.models.load_model(str(lstm_path), compile=False)
        return None

    def preprocess_data(
        self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None
    ) -> np.ndarray:
        """Preprocess data for prediction."""
        if feature_columns is None:
            feature_columns = [
                col
                for col in df.columns
                if col.startswith("sensor_") or col.startswith("op_setting_")
            ]

        X = df[feature_columns].values

        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception as e:
                logger.warning(f"Scaling failed: {e}. Using raw features.")

        return X

    def predict_classical(self, X: np.ndarray, model_name: str = "random_forest") -> np.ndarray:
        """Make predictions using classical ML model."""
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not found. Available: {list(self.models.keys())}"
            )

        model = self.models[model_name]
        return model.predict(X)

    def predict_lstm(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Make predictions using LSTM model."""
        lstm_model = self._load_lstm_model()
        if lstm_model is None:
            logger.warning("LSTM model not available")
            return None

        # Create sequences
        X_seq = []
        for i in range(len(X) - self.config.sequence_length + 1):
            X_seq.append(X[i : i + self.config.sequence_length])

        if len(X_seq) == 0:
            logger.warning("Not enough data for LSTM prediction")
            return None

        X_seq = np.array(X_seq, dtype=np.float32)
        return lstm_model.predict(X_seq, verbose=0).flatten()

    def calculate_risk_level(self, rul: float) -> str:
        """Calculate risk level based on RUL."""
        if rul < 20:
            return "CRITICAL"
        elif rul < 50:
            return "HIGH"
        elif rul < 100:
            return "MEDIUM"
        else:
            return "LOW"

    def predict_batch(
        self, input_path: str, model_name: str = "random_forest", output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run batch prediction on input data file."""
        logger.info(f"Starting batch prediction from {input_path}")

        try:
            # Load data
            df = pd.read_csv(input_path)
            logger.info(f"Loaded {len(df)} records")

            # Store metadata
            metadata_cols = ["unit_number", "time_in_cycles"] if "unit_number" in df.columns else []
            metadata = df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=df.index)

            # Preprocess
            X = self.preprocess_data(df)

            # Make predictions
            predictions = self.predict_classical(X, model_name)

            # Add predictions to results
            results_df = metadata.copy()
            results_df["predicted_rul"] = predictions
            results_df["risk_level"] = [self.calculate_risk_level(p) for p in predictions]
            results_df["prediction_timestamp"] = datetime.now().isoformat()
            results_df["model_used"] = model_name

            # Calculate statistics
            stats = {
                "total_records": len(predictions),
                "mean_rul": float(np.mean(predictions)),
                "min_rul": float(np.min(predictions)),
                "max_rul": float(np.max(predictions)),
                "std_rul": float(np.std(predictions)),
                "critical_count": int((np.array(predictions) < 20).sum()),
                "high_risk_count": int(
                    ((np.array(predictions) >= 20) & (np.array(predictions) < 50)).sum()
                ),
                "medium_risk_count": int(
                    ((np.array(predictions) >= 50) & (np.array(predictions) < 100)).sum()
                ),
                "low_risk_count": int((np.array(predictions) >= 100).sum()),
            }

            # Save predictions
            if output_path or self.config.save_predictions:
                output_dir = Path(self.config.predictions_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if output_path is None:
                    output_path = str(output_dir / f"predictions_{timestamp}.csv")

                results_df.to_csv(output_path, index=False)
                logger.info(f"Predictions saved to {output_path}")

                # Save statistics
                stats_path = str(output_dir / f"prediction_stats_{timestamp}.json")
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2)

            logger.info(f"Batch prediction completed. Processed {len(predictions)} records.")

            return {
                "predictions": results_df,
                "statistics": stats,
                "output_path": output_path,
                "model_used": model_name,
            }

        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise e

    def predict_streaming(
        self, data_generator, model_name: str = "random_forest", callback: Optional[callable] = None
    ):
        """Process streaming predictions with batching."""
        batch = []
        batch_count = 0

        for record in data_generator:
            batch.append(record)

            if len(batch) >= self.config.batch_size:
                df = pd.DataFrame(batch)
                X = self.preprocess_data(df)
                predictions = self.predict_classical(X, model_name)

                if callback:
                    callback(predictions, batch_count)

                batch_count += 1
                batch = []
                logger.info(f"Processed batch {batch_count}")

        # Process remaining records
        if batch:
            df = pd.DataFrame(batch)
            X = self.preprocess_data(df)
            predictions = self.predict_classical(X, model_name)

            if callback:
                callback(predictions, batch_count)

        logger.info(f"Streaming prediction completed. Total batches: {batch_count + 1}")

    def compare_models(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """Compare predictions from all available models."""
        logger.info("Comparing predictions across all models...")

        df = pd.read_csv(input_path)
        X = self.preprocess_data(df)

        results = pd.DataFrame()

        if "unit_number" in df.columns:
            results["unit_number"] = df["unit_number"]
        if "time_in_cycles" in df.columns:
            results["time_in_cycles"] = df["time_in_cycles"]

        for model_name in self.models.keys():
            try:
                predictions = self.predict_classical(X, model_name)
                results[f"{model_name}_rul"] = predictions
            except Exception as e:
                logger.warning(f"Failed to predict with {model_name}: {e}")

        # Calculate ensemble (average)
        rul_columns = [col for col in results.columns if col.endswith("_rul")]
        if rul_columns:
            results["ensemble_rul"] = results[rul_columns].mean(axis=1)

        if output_path:
            results.to_csv(output_path, index=False)
            logger.info(f"Model comparison saved to {output_path}")

        return results


def run_batch_prediction(
    input_path: str, model_name: str = "random_forest", output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function for batch prediction."""
    predictor = BatchPredictor()
    return predictor.predict_batch(input_path, model_name, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch Prediction CLI")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--model", default="random_forest", help="Model to use")
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    results = run_batch_prediction(args.input, args.model, args.output)
    print(f"Processed {results['statistics']['total_records']} records")
    print(f"Mean RUL: {results['statistics']['mean_rul']:.2f}")
    print(f"Critical: {results['statistics']['critical_count']}")
