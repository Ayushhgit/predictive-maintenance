"""
Training Pipeline
Orchestrates the complete ML training workflow with MLflow integration.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_evaluation import ModelEvaluator
from src.components.model_trainer import ModelTrainer
from src.constants import CONFIG_FILE_PATH, RAW_DATA_DIR, TRANSFORMED_DATA_DIR, VALIDATED_DATA_DIR
from src.utils.logger import get_logger
from src.utils.model_utils import load_yaml

logger = get_logger(__name__)


class TrainingPipeline:
    """
    End-to-end ML training pipeline with MLflow tracking.

    Orchestrates:
    1. Data Ingestion
    2. Data Validation
    3. Data Transformation
    4. Model Training
    5. Model Evaluation
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or CONFIG_FILE_PATH
        self.config = self._load_config()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mlflow_enabled = self._check_mlflow()

    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration."""
        return load_yaml(self.config_path)

    def _check_mlflow(self) -> bool:
        """Check if MLflow is available."""
        try:
            import mlflow

            return True
        except ImportError:
            logger.warning("MLflow not available. Tracking disabled.")
            return False

    def _init_mlflow(self) -> Optional[Any]:
        """Initialize MLflow tracking."""
        if not self.mlflow_enabled:
            return None

        try:
            from src.mlflow_tracking import MLflowTracker

            tracker = MLflowTracker(experiment_name="predictive-maintenance")
            return tracker
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")
            return None

    def run_data_ingestion(self) -> str:
        """Execute data ingestion step."""
        logger.info("=" * 50)
        logger.info("Step 1: Data Ingestion")
        logger.info("=" * 50)

        try:
            # Check if raw data exists
            if not Path(RAW_DATA_DIR).exists():
                logger.warning(f"Raw data not found at {RAW_DATA_DIR}")
                logger.info("Skipping ingestion - using existing data")
                return RAW_DATA_DIR

            logger.info(f"Data available at {RAW_DATA_DIR}")
            return RAW_DATA_DIR

        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise e

    def run_data_validation(self, raw_data_path: str) -> str:
        """Execute data validation step."""
        logger.info("=" * 50)
        logger.info("Step 2: Data Validation")
        logger.info("=" * 50)

        try:
            validator = DataValidation()
            validated_path = validator.initiate_data_validation(raw_data_path)
            logger.info(f"Validation completed: {validated_path}")
            return validated_path

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise e

    def run_data_transformation(self, validated_data_path: str) -> tuple:
        """Execute data transformation step."""
        logger.info("=" * 50)
        logger.info("Step 3: Data Transformation")
        logger.info("=" * 50)

        try:
            transformer = DataTransformation()
            X_train_path, X_test_path, y_train_path, y_test_path = (
                transformer.initiate_data_transformation(validated_data_path)
            )

            logger.info("Transformation completed:")
            logger.info(f"  - X_train: {X_train_path}")
            logger.info(f"  - X_test: {X_test_path}")
            logger.info(f"  - y_train: {y_train_path}")
            logger.info(f"  - y_test: {y_test_path}")

            return X_train_path, X_test_path, y_train_path, y_test_path

        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise e

    def run_model_training(
        self,
        X_train_path: str,
        X_test_path: str,
        y_train_path: str,
        y_test_path: str,
        tracker: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute model training step."""
        logger.info("=" * 50)
        logger.info("Step 4: Model Training")
        logger.info("=" * 50)

        try:
            trainer = ModelTrainer()

            # Log parameters if MLflow available
            if tracker:
                tracker.log_params(
                    {
                        "model.lstm.sequence_length": self.config["model"]["lstm"][
                            "sequence_length"
                        ],
                        "model.lstm.units": self.config["model"]["lstm"]["units"],
                        "model.lstm.dropout": self.config["model"]["lstm"]["dropout"],
                        "model.lstm.epochs": self.config["model"]["lstm"]["epochs"],
                    }
                )

            results = trainer.initiate_model_training(
                X_train_path=X_train_path,
                X_test_path=X_test_path,
                y_train_path=y_train_path,
                y_test_path=y_test_path,
            )

            logger.info(f"Training completed. Models saved to: {results['model_dir']}")
            return results

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise e

    def run_model_evaluation(
        self, X_test_path: str, y_test_path: str, tracker: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Execute model evaluation step."""
        logger.info("=" * 50)
        logger.info("Step 5: Model Evaluation")
        logger.info("=" * 50)

        try:
            evaluator = ModelEvaluator()
            results = evaluator.initiate_model_evaluation(
                X_test_path=X_test_path, y_test_path=y_test_path
            )

            # Log metrics if MLflow available
            if tracker and "report" in results:
                report = results["report"]
                if report.get("best_model") and report["best_model"].get("metrics"):
                    metrics = report["best_model"]["metrics"]
                    tracker.log_metrics(
                        {
                            "best_rmse": metrics.get("rmse", 0),
                            "best_mae": metrics.get("mae", 0),
                            "best_r2": metrics.get("r2", 0),
                        }
                    )
                    tracker.set_tag("best_model", report["best_model"].get("name", "unknown"))

            logger.info("Evaluation completed")
            if "report" in results and results["report"].get("best_model"):
                best = results["report"]["best_model"]
                logger.info(
                    f"Best model: {best.get('name')} with RMSE: {best.get('rmse', 'N/A'):.4f}"
                )

            return results

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise e

    def run(self, skip_validation: bool = False) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.

        Args:
            skip_validation: Skip data validation step

        Returns:
            Dictionary with pipeline results
        """
        logger.info("=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info(f"Run ID: {self.run_id}")
        logger.info("=" * 60)

        pipeline_results = {
            "run_id": self.run_id,
            "start_time": datetime.now().isoformat(),
            "steps": {},
        }

        tracker = None
        try:
            # Initialize MLflow
            if self.mlflow_enabled:
                tracker = self._init_mlflow()
                if tracker:
                    tracker.start_run(run_name=f"training_{self.run_id}")
                    tracker.set_tags({"pipeline": "training", "run_id": self.run_id})

            # Step 1: Data Ingestion
            raw_data_path = self.run_data_ingestion()
            pipeline_results["steps"]["ingestion"] = {"status": "success", "path": raw_data_path}

            # Step 2: Data Validation
            if skip_validation:
                validated_path = raw_data_path
                logger.info("Skipping validation as requested")
            else:
                validated_path = self.run_data_validation(raw_data_path)
            pipeline_results["steps"]["validation"] = {"status": "success", "path": validated_path}

            # Step 3: Data Transformation
            X_train_path, X_test_path, y_train_path, y_test_path = self.run_data_transformation(
                validated_path
            )
            pipeline_results["steps"]["transformation"] = {
                "status": "success",
                "paths": {
                    "X_train": X_train_path,
                    "X_test": X_test_path,
                    "y_train": y_train_path,
                    "y_test": y_test_path,
                },
            }

            # Step 4: Model Training
            training_results = self.run_model_training(
                X_train_path, X_test_path, y_train_path, y_test_path, tracker
            )
            pipeline_results["steps"]["training"] = {
                "status": "success",
                "results": training_results,
            }

            # Step 5: Model Evaluation
            evaluation_results = self.run_model_evaluation(X_test_path, y_test_path, tracker)
            pipeline_results["steps"]["evaluation"] = {
                "status": "success",
                "results": evaluation_results,
            }

            # Log artifacts if MLflow available
            if tracker:
                tracker.log_artifacts(self.config["artifacts"]["model_dir"], "models")
                tracker.log_artifacts(self.config["artifacts"]["reports_dir"], "reports")

            pipeline_results["status"] = "success"
            pipeline_results["end_time"] = datetime.now().isoformat()

            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)

        except Exception as e:
            pipeline_results["status"] = "failed"
            pipeline_results["error"] = str(e)
            pipeline_results["end_time"] = datetime.now().isoformat()
            logger.error(f"Pipeline failed: {e}")

            if tracker:
                tracker.set_tag("status", "failed")
                tracker.end_run(status="FAILED")

            raise e

        finally:
            if tracker:
                tracker.end_run()

        return pipeline_results


class BatchPredictionPipeline:
    """Pipeline for batch prediction jobs."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or CONFIG_FILE_PATH

    def run(
        self, input_path: str, model_name: str = "random_forest", output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run batch prediction pipeline."""
        from src.components.batch_prediction import BatchPredictor

        logger.info("Starting batch prediction pipeline...")

        predictor = BatchPredictor(self.config_path)
        results = predictor.predict_batch(input_path, model_name, output_path)

        logger.info("Batch prediction pipeline completed")
        return results


def run_training_pipeline(skip_validation: bool = False) -> Dict[str, Any]:
    """Convenience function to run training pipeline."""
    pipeline = TrainingPipeline()
    return pipeline.run(skip_validation=skip_validation)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training Pipeline CLI")
    parser.add_argument("--skip-validation", action="store_true", help="Skip data validation")

    args = parser.parse_args()

    results = run_training_pipeline(skip_validation=args.skip_validation)
    print(f"\nPipeline completed with status: {results['status']}")
