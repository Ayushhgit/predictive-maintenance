"""
Model Evaluation Component
Comprehensive model evaluation with visualizations and metrics reporting.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
import tensorflow as tf

from src.utils.logger import get_logger
from src.utils.model_utils import load_yaml, load_csv, create_sequences_for_regression
from src.constants import CONFIG_FILE_PATH, MODEL_DIR, LSTM_MODEL_NAME

logger = get_logger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    model_dir: str
    reports_dir: str
    sequence_length: int
    generate_plots: bool = True
    save_predictions: bool = True


class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path or CONFIG_FILE_PATH)
        self.evaluation_results = {}

    def _load_config(self, config_path: str) -> EvaluationConfig:
        """Load configuration from YAML file."""
        try:
            config_dict = load_yaml(config_path)
            return EvaluationConfig(
                model_dir=config_dict["artifacts"]["model_dir"],
                reports_dir=config_dict["artifacts"]["reports_dir"],
                sequence_length=config_dict["model"]["lstm"]["sequence_length"],
            )
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise e

    def _load_classical_model(self, model_name: str) -> Any:
        """Load a classical ML model from pickle file."""
        model_path = Path(self.config.model_dir) / f"{model_name}_model.pkl"
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None

        with open(model_path, "rb") as f:
            return pickle.load(f)

    def _load_lstm_model(self) -> Optional[tf.keras.Model]:
        """Load the LSTM model."""
        model_path = Path(self.config.model_dir) / LSTM_MODEL_NAME
        if not model_path.exists():
            logger.warning(f"LSTM model not found: {model_path}")
            return None

        return tf.keras.models.load_model(model_path, compile=False)

    def _load_scaler(self) -> Any:
        """Load the feature scaler."""
        scaler_path = Path(self.config.model_dir) / "scaler.pkl"
        if not scaler_path.exists():
            logger.warning(f"Scaler not found: {scaler_path}")
            return None

        with open(scaler_path, "rb") as f:
            return pickle.load(f)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE with handling for zero values
        mask = y_true != 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
        else:
            mape = np.inf

        # Symmetric MAPE
        smape = 100 * np.mean(
            2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
        )

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
            "smape": float(smape),
            "max_error": float(np.max(np.abs(y_true - y_pred))),
            "min_error": float(np.min(np.abs(y_true - y_pred))),
            "std_error": float(np.std(y_true - y_pred)),
        }

    def evaluate_classical_models(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all classical models."""
        logger.info("Evaluating classical models...")

        model_names = [
            "random_forest",
            "gradient_boosting",
            "linear_regression",
            "ridge",
            "lasso",
            "svr",
        ]

        results = {}
        for model_name in model_names:
            model = self._load_classical_model(model_name)
            if model is None:
                continue

            try:
                y_pred = model.predict(X_test)
                metrics = self.calculate_metrics(y_test.values, y_pred)
                results[model_name] = {"metrics": metrics, "predictions": y_pred.tolist()}
                logger.info(f"{model_name}: RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")

        return results

    def evaluate_lstm_model(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """Evaluate the LSTM model."""
        logger.info("Evaluating LSTM model...")

        model = self._load_lstm_model()
        if model is None:
            return None

        try:
            X_seq, y_seq = create_sequences_for_regression(
                X_test.values, y_test.values, self.config.sequence_length
            )

            if len(X_seq) == 0:
                logger.warning("No sequences created for LSTM evaluation")
                return None

            X_seq = np.array(X_seq, dtype=np.float32)
            y_seq = np.array(y_seq, dtype=np.float32)

            y_pred = model.predict(X_seq, verbose=0).flatten()
            metrics = self.calculate_metrics(y_seq, y_pred)

            logger.info(f"LSTM: RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")

            return {"metrics": metrics, "predictions": y_pred.tolist(), "actual": y_seq.tolist()}
        except Exception as e:
            logger.error(f"Error evaluating LSTM model: {e}")
            return None

    def generate_comparison_plot(
        self, results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None
    ) -> None:
        """Generate model comparison visualization."""
        if not results:
            logger.warning("No results to plot")
            return

        model_names = list(results.keys())
        metrics_to_plot = ["rmse", "mae", "r2"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, metric in enumerate(metrics_to_plot):
            values = [results[m]["metrics"][metric] for m in model_names]
            colors = ["#2ecc71" if metric == "r2" else "#3498db" for _ in values]
            bars = axes[idx].bar(model_names, values, color=colors, edgecolor="black")

            axes[idx].set_title(
                f"Model Comparison: {metric.upper()}", fontsize=12, fontweight="bold"
            )
            axes[idx].set_xlabel("Model", fontsize=10)
            axes[idx].set_ylabel(metric.upper(), fontsize=10)
            axes[idx].tick_params(axis="x", rotation=45)

            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[idx].annotate(
                    f"{val:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Comparison plot saved to {save_path}")

        plt.close()

    def generate_prediction_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save_dir: Optional[str] = None,
    ) -> None:
        """Generate detailed prediction analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Actual vs Predicted scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, edgecolors="black", linewidth=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot(
            [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction"
        )
        axes[0, 0].set_xlabel("Actual RUL", fontsize=11)
        axes[0, 0].set_ylabel("Predicted RUL", fontsize=11)
        axes[0, 0].set_title(f"{model_name}: Actual vs Predicted", fontsize=12, fontweight="bold")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Residual plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, edgecolors="black", linewidth=0.5)
        axes[0, 1].axhline(y=0, color="r", linestyle="--", lw=2)
        axes[0, 1].set_xlabel("Predicted RUL", fontsize=11)
        axes[0, 1].set_ylabel("Residual (Actual - Predicted)", fontsize=11)
        axes[0, 1].set_title(f"{model_name}: Residual Plot", fontsize=12, fontweight="bold")
        axes[0, 1].grid(True, alpha=0.3)

        # Residual distribution
        axes[1, 0].hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="#3498db")
        axes[1, 0].axvline(x=0, color="r", linestyle="--", lw=2)
        axes[1, 0].set_xlabel("Residual", fontsize=11)
        axes[1, 0].set_ylabel("Frequency", fontsize=11)
        axes[1, 0].set_title(f"{model_name}: Residual Distribution", fontsize=12, fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)

        # Time series comparison
        n_samples = min(200, len(y_true))
        axes[1, 1].plot(
            range(n_samples), y_true[:n_samples], label="Actual", alpha=0.8, linewidth=1.5
        )
        axes[1, 1].plot(
            range(n_samples), y_pred[:n_samples], label="Predicted", alpha=0.8, linewidth=1.5
        )
        axes[1, 1].set_xlabel("Sample Index", fontsize=11)
        axes[1, 1].set_ylabel("RUL", fontsize=11)
        axes[1, 1].set_title(
            f"{model_name}: RUL Prediction Over Time", fontsize=12, fontweight="bold"
        )
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / f"{model_name}_evaluation.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Prediction plot saved to {save_path}")

        plt.close()

    def generate_evaluation_report(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report = {"summary": {}, "detailed_metrics": {}, "best_model": None, "recommendations": []}

        if not results:
            return report

        best_rmse = float("inf")
        best_model = None

        for model_name, result in results.items():
            metrics = result["metrics"]
            report["detailed_metrics"][model_name] = metrics

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model = model_name

        report["best_model"] = {
            "name": best_model,
            "rmse": best_rmse,
            "metrics": results[best_model]["metrics"] if best_model else None,
        }

        all_rmse = [r["metrics"]["rmse"] for r in results.values()]
        all_r2 = [r["metrics"]["r2"] for r in results.values()]

        report["summary"] = {
            "total_models_evaluated": len(results),
            "avg_rmse": float(np.mean(all_rmse)),
            "avg_r2": float(np.mean(all_r2)),
            "best_rmse": float(min(all_rmse)),
            "best_r2": float(max(all_r2)),
        }

        if best_model:
            best_r2 = results[best_model]["metrics"]["r2"]
            if best_r2 > 0.9:
                report["recommendations"].append(
                    f"Excellent performance! {best_model} achieves R2 > 0.9"
                )
            elif best_r2 > 0.7:
                report["recommendations"].append(
                    f"Good performance. Consider feature engineering to improve {best_model}"
                )
            else:
                report["recommendations"].append(
                    "Consider additional feature engineering or trying ensemble methods"
                )

        return report

    def initiate_model_evaluation(self, X_test_path: str, y_test_path: str) -> Dict[str, Any]:
        """Run complete model evaluation pipeline."""
        logger.info("Starting model evaluation pipeline...")

        try:
            X_test = load_csv(X_test_path)
            y_test = load_csv(y_test_path).squeeze()

            logger.info(f"Test data loaded: X_test {X_test.shape}, y_test {y_test.shape}")

            reports_dir = Path(self.config.reports_dir)
            reports_dir.mkdir(parents=True, exist_ok=True)

            all_results = {}

            classical_results = self.evaluate_classical_models(X_test, y_test)
            all_results.update(classical_results)

            lstm_results = self.evaluate_lstm_model(X_test, y_test)
            if lstm_results:
                all_results["lstm"] = lstm_results

            if self.config.generate_plots:
                self.generate_comparison_plot(
                    all_results, save_path=str(reports_dir / "model_comparison.png")
                )

                for model_name, result in all_results.items():
                    if "predictions" in result:
                        y_pred = np.array(result["predictions"])
                        if "actual" in result:
                            y_true = np.array(result["actual"])
                        else:
                            y_true = y_test.values[: len(y_pred)]

                        self.generate_prediction_plots(
                            y_true, y_pred, model_name, save_dir=str(reports_dir)
                        )

            evaluation_report = self.generate_evaluation_report(all_results)

            report_path = reports_dir / "evaluation_report.json"
            with open(report_path, "w") as f:
                json.dump(evaluation_report, f, indent=2)
            logger.info(f"Evaluation report saved to {report_path}")

            self.evaluation_results = {"models": all_results, "report": evaluation_report}

            logger.info("Model evaluation completed successfully!")
            return self.evaluation_results

        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            raise e


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.initiate_model_evaluation(
        X_test_path="data/transformed/X_test.csv", y_test_path="data/transformed/y_test.csv"
    )
    print(json.dumps(results["report"], indent=2))
