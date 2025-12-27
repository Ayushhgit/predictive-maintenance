from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.constants import CONFIG_FILE_PATH, SCALER_NAME
from src.utils.logger import get_logger
from src.utils.model_utils import load_csv, load_yaml, save_csv, save_pickle

logger = get_logger(__name__)


@dataclass
class DataTransformationConfig:
    """
    Configuration for Data Transformation.
    """

    transformed_data_path: str
    scaler_path: str
    test_size: float
    random_state: int
    sensor_columns: List[str]
    op_setting_columns: List[str]
    window_sizes: List[int]
    lag_features: List[int]
    target_column: str


class DataTransformation:
    """
    Handles the entire data transformation pipeline for RUL prediction.
    """

    def __init__(self):
        self.config = self._load_config()
        self.scaler = StandardScaler()

    def _load_config(self) -> DataTransformationConfig:
        """Loads configuration from the YAML file."""
        try:
            config = load_yaml(CONFIG_FILE_PATH)

            # These are the columns provided by the user. They need to be configured.
            sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
            op_setting_cols = [f"op_setting_{i}" for i in range(1, 4)]

            # Update the configuration dynamically based on the new features
            config["features"]["sensor_columns"] = sensor_cols
            config["features"]["op_setting_columns"] = op_setting_cols

            return DataTransformationConfig(
                transformed_data_path=config["data"]["transformed_data_path"],
                scaler_path=f"{config['artifacts']['model_dir']}/{SCALER_NAME}",
                test_size=config["model"]["test_size"],
                random_state=config["model"]["random_state"],
                sensor_columns=sensor_cols,
                op_setting_columns=op_setting_cols,
                window_sizes=config["features"]["window_sizes"],
                lag_features=config["features"]["lag_features"],
                target_column="RUL",  # The new target is RUL
            )
        except Exception as e:
            logger.error(f"Error loading data transformation config: {e}")
            raise e

    def calculate_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Remaining Useful Life (RUL) for each unit.
        RUL is defined as the total cycles minus the current cycle count.
        """
        logger.info("Calculating Remaining Useful Life (RUL)...")
        # Find the max cycle for each unit
        max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
        # Calculate RUL
        df["RUL"] = max_cycle - df["time_in_cycles"]
        return df

    def create_lag_features(self, df: pd.DataFrame, columns_to_lag: List[str]) -> pd.DataFrame:
        """
        Creates lagged features for specified columns.
        """
        logger.info("Creating lag features...")
        df_with_lags = df.copy()
        for column in columns_to_lag:
            for lag in self.config.lag_features:
                df_with_lags[f"{column}_lag_{lag}"] = df_with_lags.groupby("unit_number")[
                    column
                ].shift(lag)

        logger.info(f"Created lag features for {len(columns_to_lag)} columns.")
        return df_with_lags

    def create_rolling_features(self, df: pd.DataFrame, columns_to_roll: List[str]) -> pd.DataFrame:
        """
        Creates rolling window features for specified columns.
        """
        logger.info("Creating rolling window features...")
        df_with_rolling = df.copy()
        for column in columns_to_roll:
            for window in self.config.window_sizes:
                df_with_rolling[f"{column}_rolling_mean_{window}"] = (
                    df_with_rolling.groupby("unit_number")[column]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                df_with_rolling[f"{column}_rolling_std_{window}"] = (
                    df_with_rolling.groupby("unit_number")[column]
                    .rolling(window=window, min_periods=1)
                    .std()
                    .reset_index(0, drop=True)
                )

        logger.info(f"Created rolling features for {len(columns_to_roll)} columns.")
        return df_with_rolling

    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates additional statistical features (not used in this specific case, but kept for extensibility).
        """
        return df

    def scale_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scales features using StandardScaler and saves the scaler.
        """
        try:
            logger.info("Scaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            save_pickle(self.scaler, self.config.scaler_path)

            logger.info(f"Feature scaling completed. Scaler saved to {self.config.scaler_path}.")
            return X_train_scaled, X_test_scaled
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            raise e

    def initiate_data_transformation(self, validated_data_path: str) -> Tuple[str, str, str, str]:
        """
        Main method to orchestrate the entire data transformation pipeline.
        """
        try:
            logger.info("Starting data transformation for RUL prediction...")

            # Load validated data
            df = load_csv(validated_data_path)

            # **New Step: Drop columns with no variance**
            columns_to_drop = [
                "sensor_6",
                "op_setting_3",
                "sensor_1",
                "sensor_5",
                "sensor_10",
                "sensor_16",
                "sensor_18",
                "sensor_19",
            ]
            df = df.drop(columns=columns_to_drop, errors="ignore")
            logger.info(f"Dropped constant-value columns: {columns_to_drop}")

            # Step 1: Calculate the RUL column
            df = self.calculate_rul(df)

            # Step 2: Create features
            remaining_sensor_cols = [
                col for col in self.config.sensor_columns if col not in columns_to_drop
            ]
            remaining_op_setting_cols = [
                col for col in self.config.op_setting_columns if col not in columns_to_drop
            ]
            columns_for_features = remaining_sensor_cols + remaining_op_setting_cols

            df = self.create_lag_features(df, columns_for_features)
            df = self.create_rolling_features(df, columns_for_features)

            # Drop rows with NaNs introduced by feature engineering
            initial_rows = len(df)
            df = df.dropna().reset_index(drop=True)
            if len(df) < initial_rows:
                logger.warning(
                    f"Dropped {initial_rows - len(df)} rows with NaN values. Remaining: {len(df)} records."
                )

            # Define features (X) and target (y)
            feature_columns = [
                col
                for col in df.columns
                if col not in ["unit_number", "time_in_cycles", self.config.target_column]
            ]
            X = df[feature_columns]
            y = df[self.config.target_column]

            # Step 3: Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )

            # Step 4: Scale features
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

            # Convert scaled arrays back to DataFrames for consistency
            X_train_df = pd.DataFrame(X_train_scaled, columns=feature_columns)
            X_test_df = pd.DataFrame(X_test_scaled, columns=feature_columns)
            y_train_df = pd.DataFrame(y_train.values, columns=[self.config.target_column])
            y_test_df = pd.DataFrame(y_test.values, columns=[self.config.target_column])

            # Save the transformed data files
            transformed_path = Path(self.config.transformed_data_path)
            save_csv(X_train_df, str(transformed_path / "X_train.csv"))
            save_csv(X_test_df, str(transformed_path / "X_test.csv"))
            save_csv(y_train_df, str(transformed_path / "y_train.csv"))
            save_csv(y_test_df, str(transformed_path / "y_test.csv"))

            logger.info("Data transformation completed successfully.")
            return (
                str(transformed_path / "X_train.csv"),
                str(transformed_path / "X_test.csv"),
                str(transformed_path / "y_train.csv"),
                str(transformed_path / "y_test.csv"),
            )

        except Exception as e:
            logger.error(f"An error occurred during the data transformation pipeline: {e}")
            raise e
