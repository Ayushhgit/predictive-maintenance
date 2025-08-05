import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
from src.utils.logger import get_logger
from src.utils.model_utils import load_yaml, load_csv, save_csv, save_pickle
from src.constants import CONFIG_FILE_PATH, SCALER_NAME, TRANSFORMED_DATA_DIR

logger = get_logger(__name__)

@dataclass
class DataTransformationConfig:
    """
    Configuration class for Data Transformation.
    Holds paths and parameters loaded from config files.
    """
    transformed_data_path: str
    scaler_path: str
    test_size: float
    random_state: int
    sensor_columns: list
    window_sizes: list
    lag_features: list
    target_column: str

class DataTransformation:
    """
    Orchestrates the data transformation pipeline, including feature engineering,
    data splitting, and feature scaling.
    """
    def __init__(self):
        """
        Initializes the DataTransformation component by loading configuration and
        setting up the scaler.
        """
        self.config = self._load_config()
        self.scaler = MinMaxScaler()
        
    def _load_config(self) -> DataTransformationConfig:
        """
        Loads and parses the configuration YAML file.
        """
        try:
            config = load_yaml(CONFIG_FILE_PATH)
            return DataTransformationConfig(
                transformed_data_path=config['data']['transformed_data_path'],
                scaler_path=f"{config['artifacts']['model_dir']}/{SCALER_NAME}",
                test_size=config['model']['test_size'],
                random_state=config['model']['random_state'],
                sensor_columns=config['features']['sensor_columns'],
                window_sizes=config['features']['window_sizes'],
                lag_features=config['features']['lag_features'],
                target_column=config['model']['target_column']
            )
        except Exception as e:
            logger.error(f"Error loading data transformation config: {e}")
            raise e
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates lagged features for specified sensor columns, grouped by equipment.
        Lag features are past values of a time series, useful for capturing temporal patterns.
        """
        try:
            logger.info("Creating lag features...")
            df_with_lags = df.copy()
            for column in self.config.sensor_columns:
                for lag in self.config.lag_features:
                    df_with_lags[f'{column}_lag_{lag}'] = df_with_lags.groupby('equipment_id')[column].shift(lag)
            
            logger.info(f"Created lag features for {len(self.config.sensor_columns)} columns with lags {self.config.lag_features}.")
            return df_with_lags
            
        except Exception as e:
            logger.error(f"Error creating lag features: {e}")
            raise e
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates rolling window features (mean, std, min, max) for sensor data.
        These features capture trends and volatility over a recent time period.
        """
        try:
            logger.info("Creating rolling window features...")
            df_with_rolling = df.copy()
            for column in self.config.sensor_columns:
                for window in self.config.window_sizes:
                    # Rolling mean
                    df_with_rolling[f'{column}_rolling_mean_{window}'] = (
                        df_with_rolling.groupby('equipment_id')[column]
                        .rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
                    )
                    # Rolling std
                    df_with_rolling[f'{column}_rolling_std_{window}'] = (
                        df_with_rolling.groupby('equipment_id')[column]
                        .rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
                    )
                    # Rolling min/max
                    df_with_rolling[f'{column}_rolling_min_{window}'] = (
                        df_with_rolling.groupby('equipment_id')[column]
                        .rolling(window=window, min_periods=1).min().reset_index(0, drop=True)
                    )
                    df_with_rolling[f'{column}_rolling_max_{window}'] = (
                        df_with_rolling.groupby('equipment_id')[column]
                        .rolling(window=window, min_periods=1).max().reset_index(0, drop=True)
                    )
            
            logger.info(f"Created rolling features for {len(self.config.sensor_columns)} columns with window sizes {self.config.window_sizes}.")
            return df_with_rolling
            
        except Exception as e:
            logger.error(f"Error creating rolling features: {e}")
            raise e
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates time-based and interaction features.
        These features can capture seasonality and complex relationships between sensors.
        """
        try:
            logger.info("Creating statistical features...")
            df_with_stats = df.copy()
            
            # Ensure 'timestamp' is datetime type for feature extraction
            if not pd.api.types.is_datetime64_any_dtype(df_with_stats['timestamp']):
                df_with_stats['timestamp'] = pd.to_datetime(df_with_stats['timestamp'])

            # Time-based features
            df_with_stats['hour'] = df_with_stats['timestamp'].dt.hour
            df_with_stats['day_of_week'] = df_with_stats['timestamp'].dt.dayofweek
            df_with_stats['month'] = df_with_stats['timestamp'].dt.month
            df_with_stats['quarter'] = df_with_stats['timestamp'].dt.quarter
            
            # Interaction features
            df_with_stats['vibration_magnitude'] = np.sqrt(
                df_with_stats['vibration_x']**2 + 
                df_with_stats['vibration_y']**2 + 
                df_with_stats['vibration_z']**2
            )
            df_with_stats['temp_pressure_ratio'] = (
                df_with_stats['temperature'] / (df_with_stats['pressure'] + 1e-8)
            )
            df_with_stats['power'] = df_with_stats['voltage'] * df_with_stats['motor_current']
            
            logger.info("Created time-based and interaction features.")
            return df_with_stats
            
        except Exception as e:
            logger.error(f"Error creating statistical features: {e}")
            raise e
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scales features using StandardScaler, fitting on the training data only
        and then transforming both training and test data.
        The fitted scaler is saved to disk for later use in prediction.
        """
        try:
            logger.info("Scaling features...")
            
            # Fit scaler on training data and transform both train and test
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Save the fitted scaler as a pickle file
            save_pickle(self.scaler, self.config.scaler_path)
            
            logger.info(f"Feature scaling completed. Scaler saved to {self.config.scaler_path}.")
            return X_train_scaled, X_test_scaled
            
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            raise e
    
    def initiate_data_transformation(self, validated_data_path: str) -> Tuple[str, str, str, str]:
        """
        Main method to orchestrate the entire data transformation pipeline.
        It loads validated data, applies all feature engineering, splits the data,
        scales the features, and saves the final datasets.
        """
        try:
            logger.info("Starting data transformation process...")
            
            # Load validated data
            df = load_csv(validated_data_path)
            logger.info(f"Loaded {len(df)} records for transformation from {validated_data_path}.")
            
            # Sort data by equipment and timestamp for time-series feature engineering
            df = df.sort_values(['equipment_id', 'timestamp']).reset_index(drop=True)
            
            # Apply all feature engineering steps
            df = self.create_lag_features(df)
            df = self.create_rolling_features(df)
            df = self.create_statistical_features(df)
            
            # Remove rows with NaN values introduced by lag/rolling features
            initial_rows = len(df)
            df = df.dropna().reset_index(drop=True)
            if len(df) < initial_rows:
                logger.warning(f"Dropped {initial_rows - len(df)} rows due to NaN values after feature engineering. Remaining: {len(df)} records.")
            
            # Prepare features (X) and target (y)
            feature_columns = [col for col in df.columns if col not in ['timestamp', 'equipment_id', self.config.target_column]]
            X = df[feature_columns]
            y = df[self.config.target_column]
            
            logger.info(f"Prepared features X shape: {X.shape}, target y shape: {y.shape}.")

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, 
                random_state=self.config.random_state, 
                stratify=y  # Stratify to ensure class balance is preserved in splits
            )
            logger.info(f"Data split into X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}.")
            
            # Scale features and save the scaler
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
            
            # Convert scaled arrays back to DataFrames for saving, preserving column names
            X_train_df = pd.DataFrame(X_train_scaled, columns=feature_columns)
            X_test_df = pd.DataFrame(X_test_scaled, columns=feature_columns)
            y_train_df = pd.DataFrame(y_train.values, columns=[self.config.target_column])
            y_test_df = pd.DataFrame(y_test.values, columns=[self.config.target_column])
            
            # Save all transformed data files to the designated directory
            save_csv(X_train_df, str(Path(self.config.transformed_data_path) / "X_train.csv"))
            save_csv(X_test_df, str(Path(self.config.transformed_data_path) / "X_test.csv"))
            save_csv(y_train_df, str(Path(self.config.transformed_data_path) / "y_train.csv"))
            save_csv(y_test_df, str(Path(self.config.transformed_data_path) / "y_test.csv"))
            
            logger.info(f"Data transformation completed. Transformed data saved to {self.config.transformed_data_path}.")
            return (
                str(Path(self.config.transformed_data_path) / "X_train.csv"), 
                str(Path(self.config.transformed_data_path) / "X_test.csv"), 
                str(Path(self.config.transformed_data_path) / "y_train.csv"), 
                str(Path(self.config.transformed_data_path) / "y_test.csv")
            )
            
        except Exception as e:
            logger.error(f"An error occurred during the data transformation pipeline: {e}")
            raise e