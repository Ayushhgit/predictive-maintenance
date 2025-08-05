import pandas as pd
from dataclasses import dataclass
import pathlib
import os
from typing import Dict, List
from src.utils.logger import get_logger
from src.utils.model_utils import load_yaml, load_csv, save_csv
from src.constants import CONFIG_FILE_PATH, SCHEMA_FILE_PATH, VALIDATED_DATA_DIR

logger = get_logger(__name__)

@dataclass
class DataValidationConfig:
    validated_data_path: str
    schema: Dict

class DataValidation:
    def __init__(self):
        self.config = self._load_config()

    def _load_config(self) -> DataValidationConfig:
        """Loads and parses the configuration and schema files."""
        try:
            config = load_yaml(CONFIG_FILE_PATH)
            schema = load_yaml(SCHEMA_FILE_PATH)
            return DataValidationConfig(
                validated_data_path=f"{config['data']['validated_data_path']}validated_data.csv",
                schema=schema
            )
        except Exception as e:
            logger.error(f"Error loading data validation config: {e}")
            raise e
        
    def validate_columns(self, df: pd.DataFrame) -> bool:
        """
        Validates if all required columns are present in the DataFrame.
        """
        try:
            required_columns = set(self.config.schema['columns'].keys())
            actual_columns = set(df.columns)

            missing_columns = required_columns - actual_columns
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            extra_columns = actual_columns - required_columns
            if extra_columns:
                logger.warning(f"Found extra columns: {extra_columns}. These will be dropped.")
            
            logger.info("All required columns are present.")
            return True
        
        except Exception as e:
            logger.error(f"Error validating columns: {e}")
            return False
        
    def validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and corrects data types based on the schema.
        """
        try:
            logger.info("Validating and correcting data types...")
            
            for column, specs in self.config.schema['columns'].items():
                if column in df.columns:
                    expected_dtype = specs['dtype']
                    
                    if expected_dtype == 'datetime64[ns]':
                        # Convert to datetime, coercing invalid dates to NaT
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    elif expected_dtype == 'float64':
                        df[column] = pd.to_numeric(df[column], errors='coerce').astype(float)
                    elif expected_dtype == 'int64':
                        # Use 'Int64' (nullable integer) for columns that might have NaNs after coercion
                        df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                    elif expected_dtype == 'object':
                        df[column] = df[column].astype(str)
            
            logger.info("Data types validated and corrected.")
            return df
            
        except Exception as e:
            logger.error(f"Error validating data types: {e}")
            raise e

    def validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates data ranges and caps outliers to the defined range.
        """
        try:
            logger.info("Validating data ranges and handling outliers...")
            
            for column, specs in self.config.schema['columns'].items():
                if column in df.columns and 'range' in specs and pd.api.types.is_numeric_dtype(df[column]):
                    min_val, max_val = specs['range']
                    
                    out_of_range_mask = (df[column] < min_val) | (df[column] > max_val)
                    num_outliers = out_of_range_mask.sum()
                    
                    if num_outliers > 0:
                        logger.warning(f"Found {num_outliers} outliers in column '{column}'. Capping values to range [{min_val}, {max_val}].")
                        df[column] = df[column].clip(lower=min_val, upper=max_val)
            
            logger.info("Data ranges validated and outliers capped.")
            return df
            
        except Exception as e:
            logger.error(f"Error validating ranges: {e}")
            raise e
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values using a forward-fill and backward-fill strategy for time-series data.
        """
        try:
            logger.info("Handling missing values...")
            
            missing_before = df.isnull().sum()
            missing_cols_before = missing_before[missing_before > 0]
            if not missing_cols_before.empty:
                logger.info(f"Missing values before handling:\\n{missing_cols_before}")
            
            # Use forward-fill and backward-fill, grouped by equipment, to maintain time series integrity
            df = df.groupby('equipment_id').ffill()
            df = df.groupby('equipment_id').bfill()
            
            # For any remaining NaNs, fill with mean or mode
            for col in df.columns:
                if df[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fill_value = df[col].mean() if not df[col].isnull().all() else 0
                        df[col] = df[col].fillna(fill_value)
                        logger.warning(f"Column '{col}' had remaining NaNs, filled with mean: {fill_value}.")
                    elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                        fill_value = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                        df[col] = df[col].fillna(fill_value)
                        logger.warning(f"Column '{col}' had remaining NaNs, filled with mode: {fill_value}.")

            logger.info("All missing values have been handled.")
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise e
    
    def save_validated_data(self, df: pd.DataFrame, output_path: str) -> str:
        """
        Saves the validated DataFrame to the specified path with enhanced error handling.
        """
        try:
            # Convert to pathlib.Path for better path handling
            validated_file_path = pathlib.Path(output_path)
            
            # Create directory structure if it doesn't exist
            validated_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Verify directory was created and is writable
            if not validated_file_path.parent.exists():
                raise IOError(f"Could not create directory: {validated_file_path.parent}")
            
            if not os.access(validated_file_path.parent, os.W_OK):
                raise IOError(f"Directory is not writable: {validated_file_path.parent}")
            
            # Save the CSV file
            logger.info(f"Saving validated data to: {validated_file_path}")
            
            # Use pandas to_csv directly with error handling as fallback
            try:
                save_csv(df, str(validated_file_path))
                logger.info(f"Successfully saved using save_csv utility function.")
            except Exception as save_util_error:
                logger.warning(f"save_csv utility failed: {save_util_error}. Falling back to pandas to_csv.")
                df.to_csv(validated_file_path, index=False)
                logger.info(f"Successfully saved using pandas to_csv fallback.")
            
            # Verify the file was actually created and has content
            if not validated_file_path.exists():
                raise IOError(f"File was not created: {validated_file_path}")
            
            file_size = validated_file_path.stat().st_size
            if file_size == 0:
                raise IOError(f"File was created but is empty: {validated_file_path}")
            
            logger.info(f"Validated data successfully saved to {validated_file_path} (Size: {file_size} bytes)")
            return str(validated_file_path)
            
        except Exception as e:
            logger.error(f"Error saving validated data: {e}")
            raise e
        
    def initiate_data_validation(self, raw_data_path: str) -> str:
        """
        Orchestrates the entire data validation process.
        """
        try:
            logger.info("Starting data validation process...")
            
            # Load raw data
            df = load_csv(raw_data_path)
            logger.info(f"Loaded {len(df)} records for validation from {raw_data_path}.")
            
            # 1. Validate columns
            if not self.validate_columns(df):
                logger.error("Data validation failed: Missing required columns.")
                raise ValueError("Missing required columns in raw data.")
            
            # Drop extra columns to ensure a clean, consistent schema
            required_columns = set(self.config.schema['columns'].keys())
            df = df[list(required_columns)]

            # 2. Validate and correct data types
            df = self.validate_data_types(df)
            
            # 3. Validate ranges and handle outliers
            df = self.validate_ranges(df)
            
            # 4. Handle missing values
            df = self.handle_missing_values(df)
            
            # 5. Save validated data with enhanced error handling
            validated_file_path = self.save_validated_data(df, self.config.validated_data_path)
            
            logger.info(f"Data validation completed successfully. Final validated dataset contains {len(df)} records.")
            return validated_file_path
            
        except Exception as e:
            logger.error(f"An error occurred during the data validation pipeline: {e}")
            raise e