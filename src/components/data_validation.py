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
            
            # Debug: Print the schema structure
            logger.info(f"Loaded schema structure: {schema}")
            logger.info(f"Schema keys: {list(schema.keys()) if isinstance(schema, dict) else 'Not a dictionary'}")
            
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
            # Debug: Print what we're working with
            logger.info(f"Schema type: {type(self.config.schema)}")
            logger.info(f"Schema content: {self.config.schema}")
            
            # Handle different possible schema structures
            required_columns = None
            
            if 'columns' in self.config.schema:
                if isinstance(self.config.schema['columns'], list):
                    required_columns = set(self.config.schema['columns'])
                    logger.info("Using list format for columns")
                elif isinstance(self.config.schema['columns'], dict):
                    required_columns = set(self.config.schema['columns'].keys())
                    logger.info("Using dictionary format for columns")
                else:
                    logger.error(f"Unexpected columns format: {type(self.config.schema['columns'])}")
                    return False
            elif 'schema' in self.config.schema and 'columns' in self.config.schema['schema']:
                # Handle nested schema structure
                columns_data = self.config.schema['schema']['columns']
                if isinstance(columns_data, list):
                    required_columns = set(columns_data)
                    logger.info("Using nested list format for columns")
                elif isinstance(columns_data, dict):
                    required_columns = set(columns_data.keys())
                    logger.info("Using nested dictionary format for columns")
            else:
                logger.error(f"Could not find 'columns' in schema. Available keys: {list(self.config.schema.keys())}")
                return False
            
            actual_columns = set(df.columns)
            logger.info(f"Required columns: {required_columns}")
            logger.info(f"Actual columns: {actual_columns}")

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
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
        
    def validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and corrects data types based on the schema.
        """
        try:
            logger.info("Validating and correcting data types...")
            
            # Get column specifications
            columns_spec = None
            if 'columns' in self.config.schema:
                columns_spec = self.config.schema['columns']
            elif 'schema' in self.config.schema and 'columns' in self.config.schema['schema']:
                columns_spec = self.config.schema['schema']['columns']
            
            # Check if schema has column specifications (dictionary format)
            if isinstance(columns_spec, dict):
                for column, specs in columns_spec.items():
                    if column in df.columns and isinstance(specs, dict) and 'dtype' in specs:
                        expected_dtype = specs['dtype']
                        
                        if expected_dtype == 'datetime64[ns]':
                            df[column] = pd.to_datetime(df[column], errors='coerce')
                        elif expected_dtype == 'float64':
                            df[column] = pd.to_numeric(df[column], errors='coerce').astype(float)
                        elif expected_dtype == 'int64':
                            df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                        elif expected_dtype == 'object':
                            df[column] = df[column].astype(str)
            else:
                # If columns is a list, apply default data type inference
                logger.info("Schema columns is a list format. Applying default data type inference.")
                for col in df.columns:
                    if col.startswith('sensor_') or col.startswith('op_setting_'):
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif col in ['unit_number', 'time_in_cycles']:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            
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
            
            # Get column specifications
            columns_spec = None
            if 'columns' in self.config.schema:
                columns_spec = self.config.schema['columns']
            elif 'schema' in self.config.schema and 'columns' in self.config.schema['schema']:
                columns_spec = self.config.schema['schema']['columns']
            
            # Check if schema has column specifications (dictionary format)
            if isinstance(columns_spec, dict):
                for column, specs in columns_spec.items():
                    if (column in df.columns and isinstance(specs, dict) and 
                        'range' in specs and pd.api.types.is_numeric_dtype(df[column])):
                        min_val, max_val = specs['range']
                        
                        out_of_range_mask = (df[column] < min_val) | (df[column] > max_val)
                        num_outliers = out_of_range_mask.sum()
                        
                        if num_outliers > 0:
                            logger.warning(f"Found {num_outliers} outliers in column '{column}'. Capping values to range [{min_val}, {max_val}].")
                            df[column] = df[column].clip(lower=min_val, upper=max_val)
            else:
                logger.info("Schema columns is a list format. Skipping range validation.")
            
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
            
            # Check if unit_number column exists for grouping
            if 'unit_number' in df.columns:
                logger.info(f"unit_number column dtype: {df['unit_number'].dtype}")
                logger.info(f"unit_number has NaN values: {df['unit_number'].isnull().any()}")
                
                # Handle nullable integer types that might cause groupby issues
                if df['unit_number'].dtype == 'Int64':
                    # Convert to regular int if no NaN values, otherwise handle NaNs first
                    if not df['unit_number'].isnull().any():
                        df['unit_number'] = df['unit_number'].astype('int64')
                    else:
                        # Fill NaN values in unit_number before groupby
                        df['unit_number'] = df['unit_number'].fillna(df['unit_number'].mode()[0] if not df['unit_number'].mode().empty else 1)
                        df['unit_number'] = df['unit_number'].astype('int64')
                
                # Perform groupby operations
                try:
                    df = df.groupby('unit_number').ffill()
                    df = df.groupby('unit_number').bfill()
                    logger.info("Successfully applied grouped forward-fill and backward-fill.")
                except Exception as groupby_error:
                    logger.warning(f"Groupby operation failed: {groupby_error}. Using simple ffill/bfill without grouping.")
                    df = df.ffill()
                    df = df.bfill()
            else:
                logger.warning("unit_number column not found. Using simple ffill/bfill without grouping.")
                df = df.ffill()
                df = df.bfill()
            
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
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise e
    
    def save_validated_data(self, df: pd.DataFrame, output_path: str) -> str:
        """
        Saves the validated DataFrame to the specified path with enhanced error handling.
        """
        try:
            validated_file_path = pathlib.Path(output_path)
            validated_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not validated_file_path.parent.exists():
                raise IOError(f"Could not create directory: {validated_file_path.parent}")
            
            if not os.access(validated_file_path.parent, os.W_OK):
                raise IOError(f"Directory is not writable: {validated_file_path.parent}")
            
            logger.info(f"Saving validated data to: {validated_file_path}")
            
            try:
                save_csv(df, str(validated_file_path))
                logger.info(f"Successfully saved using save_csv utility function.")
            except Exception as save_util_error:
                logger.warning(f"save_csv utility failed: {save_util_error}. Falling back to pandas to_csv.")
                df.to_csv(validated_file_path, index=False)
                logger.info(f"Successfully saved using pandas to_csv fallback.")
            
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
            logger.info(f"DataFrame columns: {list(df.columns)}")
            
            # 1. Validate columns
            if not self.validate_columns(df):
                logger.error("Data validation failed: Missing required columns.")
                raise ValueError("Missing required columns in raw data.")
            
            # Drop extra columns to ensure a clean, consistent schema while preserving order
            required_columns_ordered = None
            if 'columns' in self.config.schema:
                if isinstance(self.config.schema['columns'], list):
                    required_columns_ordered = self.config.schema['columns']
                else:
                    required_columns_ordered = list(self.config.schema['columns'].keys())
            elif 'schema' in self.config.schema and 'columns' in self.config.schema['schema']:
                columns_data = self.config.schema['schema']['columns']
                if isinstance(columns_data, list):
                    required_columns_ordered = columns_data
                else:
                    required_columns_ordered = list(columns_data.keys())
            
            if required_columns_ordered:
                # Filter to only include columns that exist in the DataFrame, preserving order
                existing_columns = [col for col in required_columns_ordered if col in df.columns]
                df = df[existing_columns]
                logger.info(f"Reordered DataFrame columns to match schema: {existing_columns}")

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