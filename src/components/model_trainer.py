import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from src.constants import CONFIG_FILE_PATH, LSTM_MODEL_NAME
from src.utils.logger import get_logger
from src.utils.model_utils import create_sequences_for_regression, load_csv, load_yaml, save_pickle

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Base configuration for model parameters."""

    name: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    use_grid_search: bool = False
    grid_search_params: Dict[str, List] = field(default_factory=dict)


@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""

    units: int = 128
    dropout: float = 0.3
    epochs: int = 100
    batch_size: int = 32
    sequence_length: int = 10
    learning_rate: float = 0.001
    layers: int = 2
    l2_reg: float = 0.01


@dataclass
class ModelTrainerConfig:
    """Configuration class for Model Trainer."""

    model_dir: str
    lstm_config: LSTMConfig
    target_column: str
    sequence_length: int
    validation_split: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    enable_hyperparameter_tuning: bool = False
    save_training_metrics: bool = True


class BaseModelTrainer(ABC):
    """Abstract base class for model trainers."""

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train the model."""
        pass

    @abstractmethod
    def save(self, model: Any, save_path: str) -> None:
        """Save the trained model."""
        pass


class ClassicalModelTrainer(BaseModelTrainer):
    """Trainer for classical ML models."""

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model_configs = self._get_model_configs()

    def _get_model_configs(self) -> Dict[str, ModelConfig]:
        """Define configurations for classical models."""
        return {
            "random_forest": ModelConfig(
                name="RandomForestRegressor",
                hyperparameters={
                    "n_estimators": 100,
                    "random_state": self.config.random_state,
                    "n_jobs": self.config.n_jobs,
                },
                use_grid_search=self.config.enable_hyperparameter_tuning,
                grid_search_params={
                    "n_estimators": [50, 100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5, 10],
                },
            ),
            "gradient_boosting": ModelConfig(
                name="GradientBoostingRegressor",
                hyperparameters={"n_estimators": 100, "random_state": self.config.random_state},
                use_grid_search=self.config.enable_hyperparameter_tuning,
                grid_search_params={
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.05, 0.1, 0.15],
                    "max_depth": [3, 5, 7],
                },
            ),
            "linear_regression": ModelConfig(
                name="LinearRegression", hyperparameters={"n_jobs": self.config.n_jobs}
            ),
            "ridge": ModelConfig(
                name="Ridge",
                hyperparameters={"random_state": self.config.random_state},
                use_grid_search=self.config.enable_hyperparameter_tuning,
                grid_search_params={"alpha": [0.1, 1.0, 10.0, 100.0]},
            ),
            "lasso": ModelConfig(
                name="Lasso",
                hyperparameters={"random_state": self.config.random_state},
                use_grid_search=self.config.enable_hyperparameter_tuning,
                grid_search_params={"alpha": [0.1, 1.0, 10.0, 100.0]},
            ),
            "svr": ModelConfig(
                name="SVR",
                hyperparameters={"kernel": "rbf"},
                use_grid_search=self.config.enable_hyperparameter_tuning,
                grid_search_params={"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.1, 1]},
            ),
        }

    def _create_model(self, model_name: str, config: ModelConfig) -> Any:
        """Create a model instance based on configuration."""
        model_classes = {
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "SVR": SVR,
        }

        model_class = model_classes.get(config.name)
        if not model_class:
            raise ValueError(f"Unknown model: {config.name}")

        return model_class(**config.hyperparameters)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train classical regression models."""
        try:
            logger.info("Training classical regression models...")
            trained_models = {}
            training_metrics = {}

            for model_name, config in self.model_configs.items():
                logger.info(f"Training {model_name}...")

                base_model = self._create_model(model_name, config)

                if config.use_grid_search and config.grid_search_params:
                    logger.info(f"Performing grid search for {model_name}...")
                    model = GridSearchCV(
                        base_model,
                        config.grid_search_params,
                        cv=5,
                        scoring="neg_mean_squared_error",
                        n_jobs=self.config.n_jobs,
                    )
                    model.fit(X_train, y_train)
                    logger.info(f"Best parameters for {model_name}: {model.best_params_}")
                    training_metrics[model_name] = {
                        "best_params": model.best_params_,
                        "best_score": model.best_score_,
                    }
                else:
                    model = base_model
                    model.fit(X_train, y_train)

                trained_models[model_name] = model

                # Calculate training metrics
                y_pred = model.predict(X_train)
                metrics = self._calculate_metrics(y_train, y_pred)
                training_metrics[model_name] = {**training_metrics.get(model_name, {}), **metrics}

                logger.info(
                    f"{model_name} training completed. MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}"
                )

            return trained_models, training_metrics

        except Exception as e:
            logger.error(f"Error training classical models: {e}")
            raise e

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

    def save(self, models: Dict[str, Any], save_dir: str) -> None:
        """Save trained classical models."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for name, model in models.items():
            model_path = save_path / f"{name}_model.pkl"
            save_pickle(model, str(model_path))
            logger.info(f"{name} model saved to {model_path}")


class LSTMModelTrainer(BaseModelTrainer):
    """Trainer for LSTM models."""

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.lstm_config = config.lstm_config

    def build_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build and compile an enhanced LSTM model."""
        try:
            inputs = Input(shape=input_shape, name="lstm_input")
            x = inputs

            # Add multiple LSTM layers
            for i in range(self.lstm_config.layers):
                return_sequences = i < self.lstm_config.layers - 1
                x = LSTM(
                    max(self.lstm_config.units // (2**i), 16),
                    return_sequences=return_sequences,
                    kernel_regularizer=l2(self.lstm_config.l2_reg),
                    name=f"lstm_layer_{i+1}",
                )(x)
                x = BatchNormalization(name=f"batch_norm_{i+1}")(x)
                x = Dropout(self.lstm_config.dropout, name=f"dropout_{i+1}")(x)

            # Dense layers
            x = Dense(
                64,
                activation="relu",
                kernel_regularizer=l2(self.lstm_config.l2_reg),
                name="dense_1",
            )(x)
            x = Dropout(self.lstm_config.dropout, name="dropout_dense_1")(x)
            x = Dense(
                32,
                activation="relu",
                kernel_regularizer=l2(self.lstm_config.l2_reg),
                name="dense_2",
            )(x)
            x = Dropout(self.lstm_config.dropout / 2, name="dropout_dense_2")(x)

            # Output layer
            outputs = Dense(1, name="output")(x)

            model = Model(inputs=inputs, outputs=outputs, name="lstm_rul_model")

            # Compile with custom metrics
            model.compile(
                optimizer=Adam(learning_rate=self.lstm_config.learning_rate),
                loss="mse",
                metrics=["mae", "mse", self._root_mean_squared_error],
            )

            return model

        except Exception as e:
            logger.error(f"Error building LSTM model: {e}")
            raise e

    @staticmethod
    def _root_mean_squared_error(y_true, y_pred):
        """Custom RMSE metric for TensorFlow."""
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[tf.keras.Model, Dict[str, Any]]:
        """Train the LSTM model with enhanced features."""
        try:
            logger.info("Training LSTM model...")

            # Create sequences
            X_seq, y_seq = create_sequences_for_regression(
                X_train.values, y_train.values, self.lstm_config.sequence_length
            )

            if len(X_seq) == 0:
                raise ValueError("No sequences created. Check sequence length and data size.")

            logger.info(f"Created sequences: X_seq shape {X_seq.shape}, y_seq shape {y_seq.shape}")

            X_seq = np.array(X_seq, dtype=np.float32)
            y_seq = np.array(y_seq, dtype=np.float32)

            # Split for training and validation
            split_idx = int(len(X_seq) * (1 - self.config.validation_split))
            X_train_seq, X_val_seq = X_seq[:split_idx], X_seq[split_idx:]
            y_train_seq, y_val_seq = y_seq[:split_idx], y_seq[split_idx:]

            # Build model
            input_shape = (self.lstm_config.sequence_length, X_train.shape[1])
            model = self.build_lstm_model(input_shape)

            # Enhanced callbacks
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
                ),
                ModelCheckpoint(
                    filepath=str(Path(self.config.model_dir) / LSTM_MODEL_NAME),
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=1,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7, verbose=1
                ),
            ]

            # Train the model
            history = model.fit(
                X_train_seq,
                y_train_seq,
                epochs=self.lstm_config.epochs,
                batch_size=self.lstm_config.batch_size,
                validation_data=(X_val_seq, y_val_seq),
                callbacks=callbacks,
                verbose=1,
                shuffle=True,
            )

            # Calculate final metrics
            final_metrics = {
                "final_train_loss": float(history.history["loss"][-1]),
                "final_val_loss": float(history.history["val_loss"][-1]),
                "final_train_mae": float(history.history["mae"][-1]),
                "final_val_mae": float(history.history["val_mae"][-1]),
                "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
                "total_epochs": len(history.history["loss"]),
            }

            # Save training history plot
            plot_save_path = str(Path(self.config.model_dir) / "lstm_training_history.png")

            logger.info(f"LSTM model training completed. Best epoch: {final_metrics['best_epoch']}")
            return model, final_metrics

        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            raise e

    def save(self, model: tf.keras.Model, save_path: str) -> None:
        """Save the trained LSTM model."""
        model.save(save_path)
        logger.info(f"LSTM model saved to {save_path}")


class ModelTrainer:
    """Enhanced model trainer with modular design and comprehensive features."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path or CONFIG_FILE_PATH)
        self.classical_trainer = ClassicalModelTrainer(self.config)
        self.lstm_trainer = LSTMModelTrainer(self.config)

    def _load_config(self, config_path: str) -> ModelTrainerConfig:
        """Load and validate configuration."""
        try:
            config_dict = load_yaml(config_path)

            # Create LSTM config
            lstm_config = LSTMConfig(**config_dict.get("model", {}).get("lstm", {}))

            return ModelTrainerConfig(
                model_dir=config_dict["artifacts"]["model_dir"],
                lstm_config=lstm_config,
                target_column=config_dict["model"]["target_column"],
                sequence_length=lstm_config.sequence_length,
                validation_split=config_dict.get("model", {}).get("validation_split", 0.2),
                random_state=config_dict.get("model", {}).get("random_state", 42),
                n_jobs=config_dict.get("model", {}).get("n_jobs", -1),
                enable_hyperparameter_tuning=config_dict.get("model", {}).get(
                    "enable_hyperparameter_tuning", False
                ),
                save_training_metrics=config_dict.get("model", {}).get(
                    "save_training_metrics", True
                ),
            )
        except Exception as e:
            logger.error(f"Error loading model trainer config: {e}")
            raise e

    def _save_training_metrics(
        self, metrics: Dict[str, Any], filename: str = "training_metrics.json"
    ) -> None:
        """Save training metrics to JSON file."""
        if not self.config.save_training_metrics:
            return

        try:
            metrics_path = Path(self.config.model_dir) / filename
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"Training metrics saved to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving training metrics: {e}")

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train all models (classical and LSTM)."""
        try:
            logger.info("Starting comprehensive model training...")

            # Ensure model directory exists
            Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)

            all_trained_models = {}
            all_metrics = {}

            # Train classical models
            classical_models, classical_metrics = self.classical_trainer.train(X_train, y_train)
            self.classical_trainer.save(classical_models, self.config.model_dir)

            all_trained_models.update(classical_models)
            all_metrics["classical"] = classical_metrics

            # Train LSTM model
            lstm_model, lstm_metrics = self.lstm_trainer.train(X_train, y_train)
            lstm_save_path = str(Path(self.config.model_dir) / "lstm_model.h5")
            self.lstm_trainer.save(lstm_model, lstm_save_path)

            all_trained_models["lstm"] = lstm_model
            all_metrics["lstm"] = lstm_metrics

            # Save comprehensive metrics
            self._save_training_metrics(all_metrics)

            logger.info(
                f"All model training completed successfully. Trained {len(all_trained_models)} models."
            )
            return {
                "models": all_trained_models,
                "metrics": all_metrics,
                "model_dir": self.config.model_dir,
            }

        except Exception as e:
            logger.error(f"Error in comprehensive model training: {e}")
            raise e

    def initiate_model_training(
        self, X_train_path: str, X_test_path: str, y_train_path: str, y_test_path: str
    ) -> Dict[str, Any]:
        """Initiate the complete model training pipeline."""
        try:
            logger.info("Initiating model training pipeline...")

            # Load training data
            X_train = load_csv(X_train_path)
            y_train = load_csv(y_train_path).squeeze()

            # Validate data
            if X_train.empty or y_train.empty:
                raise ValueError("Training data is empty")

            if len(X_train) != len(y_train):
                raise ValueError("X_train and y_train have different lengths")

            logger.info(f"Training data loaded: X_train {X_train.shape}, y_train {y_train.shape}")

            # Train all models
            results = self.train_all_models(X_train, y_train)

            logger.info("Model training pipeline completed successfully!")
            return results

        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise e
