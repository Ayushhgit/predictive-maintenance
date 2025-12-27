"""
MLflow Experiment Tracking and Model Registry
Provides comprehensive ML experiment tracking, model versioning, and registry management.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from src.constants import CONFIG_FILE_PATH
from src.utils.logger import get_logger
from src.utils.model_utils import load_yaml

logger = get_logger(__name__)


class MLflowTracker:
    """MLflow experiment tracking and model registry management."""

    def __init__(
        self,
        experiment_name: str = "predictive-maintenance",
        tracking_uri: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "mlruns")
        self.config = self._load_config(config_path or CONFIG_FILE_PATH)
        self.client = None
        self._setup_mlflow()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration."""
        try:
            return load_yaml(config_path)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}

    def _setup_mlflow(self) -> None:
        """Initialize MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self.client = MlflowClient()

            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    self.experiment_name,
                    tags={"project": "predictive-maintenance", "version": "1.0"},
                )

            logger.info(f"MLflow initialized with experiment: {self.experiment_name}")
            logger.info(f"Tracking URI: {self.tracking_uri}")

        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            raise e

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        default_tags = {
            "timestamp": datetime.now().isoformat(),
            "environment": os.getenv("ENVIRONMENT", "development"),
        }

        if tags:
            default_tags.update(tags)

        return mlflow.start_run(run_name=run_name, tags=default_tags, nested=nested)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        try:
            # Flatten nested dictionaries
            flat_params = self._flatten_dict(params)

            for key, value in flat_params.items():
                # MLflow has limits on parameter value length
                str_value = str(value)[:250]
                mlflow.log_param(key, str_value)

            logger.debug(f"Logged {len(flat_params)} parameters")
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    mlflow.log_metric(key, value, step=step)

            logger.debug(f"Logged {len(metrics)} metrics")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

    def log_model_sklearn(
        self,
        model: Any,
        artifact_path: str,
        signature: Optional[Any] = None,
        input_example: Optional[pd.DataFrame] = None,
        registered_model_name: Optional[str] = None,
    ) -> None:
        """Log sklearn model to MLflow."""
        try:
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
            )
            logger.info(f"Logged sklearn model: {artifact_path}")
        except Exception as e:
            logger.error(f"Error logging sklearn model: {e}")
            raise e

    def log_model_tensorflow(
        self,
        model: Any,
        artifact_path: str,
        signature: Optional[Any] = None,
        input_example: Optional[np.ndarray] = None,
        registered_model_name: Optional[str] = None,
    ) -> None:
        """Log TensorFlow/Keras model to MLflow."""
        try:
            mlflow.tensorflow.log_model(
                model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
            )
            logger.info(f"Logged TensorFlow model: {artifact_path}")
        except Exception as e:
            logger.error(f"Error logging TensorFlow model: {e}")
            raise e

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact file to MLflow."""
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Error logging artifact: {e}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log directory of artifacts to MLflow."""
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.info(f"Logged artifacts from: {local_dir}")
        except Exception as e:
            logger.error(f"Error logging artifacts: {e}")

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Log dictionary as JSON artifact."""
        try:
            mlflow.log_dict(dictionary, artifact_file)
            logger.debug(f"Logged dict artifact: {artifact_file}")
        except Exception as e:
            logger.error(f"Error logging dict: {e}")

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        """Log matplotlib figure to MLflow."""
        try:
            mlflow.log_figure(figure, artifact_file)
            logger.debug(f"Logged figure: {artifact_file}")
        except Exception as e:
            logger.error(f"Error logging figure: {e}")

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run."""
        mlflow.set_tag(key, value)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set multiple tags on the current run."""
        mlflow.set_tags(tags)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        mlflow.end_run(status=status)
        logger.info(f"MLflow run ended with status: {status}")

    def register_model(
        self, model_uri: str, name: str, tags: Optional[Dict[str, str]] = None
    ) -> Any:
        """Register a model in the MLflow Model Registry."""
        try:
            result = mlflow.register_model(model_uri, name)

            if tags:
                self.client.set_model_version_tag(
                    name=name, version=result.version, key="tags", value=json.dumps(tags)
                )

            logger.info(f"Registered model: {name} version {result.version}")
            return result
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise e

    def transition_model_stage(self, name: str, version: int, stage: str) -> None:
        """Transition model version to a new stage."""
        valid_stages = ["None", "Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage. Must be one of: {valid_stages}")

        try:
            self.client.transition_model_version_stage(name=name, version=version, stage=stage)
            logger.info(f"Transitioned {name} v{version} to {stage}")
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            raise e

    def get_latest_model_version(
        self, name: str, stages: Optional[List[str]] = None
    ) -> Optional[Any]:
        """Get the latest version of a registered model."""
        try:
            stages = stages or ["Production", "Staging", "None"]
            versions = self.client.get_latest_versions(name, stages)

            if versions:
                return max(versions, key=lambda v: int(v.version))
            return None
        except Exception as e:
            logger.error(f"Error getting latest model version: {e}")
            return None

    def load_model(self, model_uri: str) -> Any:
        """Load a model from MLflow."""
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

    def get_run_metrics(self, run_id: str) -> Dict[str, float]:
        """Get metrics from a specific run."""
        try:
            run = self.client.get_run(run_id)
            return run.data.metrics
        except Exception as e:
            logger.error(f"Error getting run metrics: {e}")
            return {}

    def search_runs(
        self, filter_string: str = "", order_by: Optional[List[str]] = None, max_results: int = 100
    ) -> pd.DataFrame:
        """Search for runs matching criteria."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return pd.DataFrame()

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                order_by=order_by or ["metrics.rmse ASC"],
                max_results=max_results,
            )
            return runs
        except Exception as e:
            logger.error(f"Error searching runs: {e}")
            return pd.DataFrame()

    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare metrics across multiple runs."""
        comparison_data = []

        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                row = {
                    "run_id": run_id,
                    "run_name": run.info.run_name,
                    **run.data.params,
                    **run.data.metrics,
                }
                comparison_data.append(row)
            except Exception as e:
                logger.warning(f"Could not fetch run {run_id}: {e}")

        return pd.DataFrame(comparison_data)

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


def get_tracker(
    experiment_name: str = "predictive-maintenance", tracking_uri: Optional[str] = None
) -> MLflowTracker:
    """Factory function to get MLflow tracker instance."""
    return MLflowTracker(experiment_name=experiment_name, tracking_uri=tracking_uri)


if __name__ == "__main__":
    # Example usage
    tracker = get_tracker()

    with tracker.start_run(run_name="test_run"):
        tracker.log_params({"learning_rate": 0.001, "epochs": 100})
        tracker.log_metrics({"rmse": 15.5, "mae": 10.2, "r2": 0.85})
        tracker.set_tag("model_type", "lstm")

    print("MLflow tracking test completed!")
