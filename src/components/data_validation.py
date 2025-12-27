import os
import pathlib
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src.constants import CONFIG_FILE_PATH, SCHEMA_FILE_PATH, VALIDATED_DATA_DIR
from src.utils.logger import get_logger
from src.utils.model_utils import load_csv, load_yaml, save_csv

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
            logger.info(
                f"Schema keys: {list(schema.keys()) if isinstance(schema, dict) else 'Not a dictionary'}"
            )

            return DataValidationConfig(
                validated_data_path=f"{config['data']['validated_data_path']}validated_data.csv",
                schema=schema,
            )
        except Exception as e:
            logger.error(f"Error loading data validation config: {e}")
            raise e

    def debug_columns_at_each_step(self, df: pd.DataFrame, step_name: str):
        """Debug helper to track columns at each step"""
        logger.info(f"=== STEP: {step_name} ===")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Columns ({len(df.columns)}): {list(df.columns)}")

        # Check specifically for unit-related columns
        unit_cols = [col for col in df.columns if "unit" in col.lower()]
        if unit_cols:
            logger.info(f"Unit-related columns found: {unit_cols}")
            for col in unit_cols:
                logger.info(f"  {col}: unique values = {df[col].nunique()}")
        else:
            logger.warning("NO unit-related columns found!")

        # Check for time-related columns
        time_cols = [
            col for col in df.columns if any(word in col.lower() for word in ["time", "cycle"])
        ]
        if time_cols:
            logger.info(f"Time-related columns found: {time_cols}")
        else:
            logger.warning("NO time-related columns found!")

        logger.info("=" * (len(step_name) + 12))

    def validate_columns(self, df: pd.DataFrame) -> bool:
        """
        Validates if all required columns are present in the DataFrame.
        """
        try:
            self.debug_columns_at_each_step(df, "BEFORE COLUMN VALIDATION")

            # Debug: Print what we're working with
            logger.info(f"Schema type: {type(self.config.schema)}")
            logger.info(f"Schema content: {self.config.schema}")

            # Handle different possible schema structures
            required_columns = None

            if "columns" in self.config.schema:
                if isinstance(self.config.schema["columns"], list):
                    required_columns = set(self.config.schema["columns"])
                    logger.info("Using list format for columns")
                elif isinstance(self.config.schema["columns"], dict):
                    required_columns = set(self.config.schema["columns"].keys())
                    logger.info("Using dictionary format for columns")
                else:
                    logger.error(
                        f"Unexpected columns format: {type(self.config.schema['columns'])}"
                    )
                    return False
            elif "schema" in self.config.schema and "columns" in self.config.schema["schema"]:
                # Handle nested schema structure
                columns_data = self.config.schema["schema"]["columns"]
                if isinstance(columns_data, list):
                    required_columns = set(columns_data)
                    logger.info("Using nested list format for columns")
                elif isinstance(columns_data, dict):
                    required_columns = set(columns_data.keys())
                    logger.info("Using nested dictionary format for columns")
            else:
                logger.error(
                    f"Could not find 'columns' in schema. Available keys: {list(self.config.schema.keys())}"
                )
                return False

            logger.info(f"Original required columns from schema: {required_columns}")

            # FORCE ADD unit_number if it exists in DataFrame
            actual_columns = set(df.columns)
            if "unit_number" in actual_columns:
                logger.info("FORCING unit_number to be included in required columns")
                required_columns.add("unit_number")

            if "time_in_cycles" in actual_columns:
                logger.info("FORCING time_in_cycles to be included in required columns")
                required_columns.add("time_in_cycles")

            # Also check for alternative columns
            unit_identifier_alternatives = {"unit", "unit_id", "engine_id", "id"}
            for alt_col in unit_identifier_alternatives:
                if alt_col in actual_columns:
                    required_columns.add(alt_col)
                    logger.info(f"Added alternative unit identifier to schema: {alt_col}")

            time_identifier_alternatives = {"cycle", "time", "cycles", "time_cycle"}
            for alt_col in time_identifier_alternatives:
                if alt_col in actual_columns:
                    required_columns.add(alt_col)
                    logger.info(f"Added alternative time identifier to schema: {alt_col}")

            logger.info(f"Final required columns: {required_columns}")
            logger.info(f"Actual columns: {actual_columns}")

            missing_columns = required_columns - actual_columns
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False

            extra_columns = actual_columns - required_columns
            if extra_columns:
                logger.warning(f"Found extra columns that will be kept: {extra_columns}")

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
            self.debug_columns_at_each_step(df, "BEFORE DATA TYPE VALIDATION")

            logger.info("Validating and correcting data types...")

            # Get column specifications
            columns_spec = None
            if "columns" in self.config.schema:
                columns_spec = self.config.schema["columns"]
            elif "schema" in self.config.schema and "columns" in self.config.schema["schema"]:
                columns_spec = self.config.schema["schema"]["columns"]

            # Check if schema has column specifications (dictionary format)
            if isinstance(columns_spec, dict):
                for column, specs in columns_spec.items():
                    if column in df.columns and isinstance(specs, dict) and "dtype" in specs:
                        expected_dtype = specs["dtype"]
                        logger.info(f"Converting {column} to {expected_dtype}")

                        if expected_dtype == "datetime64[ns]":
                            df[column] = pd.to_datetime(df[column], errors="coerce")
                        elif expected_dtype == "float64":
                            df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
                        elif expected_dtype == "int64":
                            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
                        elif expected_dtype == "object":
                            df[column] = df[column].astype(str)
            else:
                # If columns is a list, apply default data type inference
                logger.info(
                    "Schema columns is a list format. Applying default data type inference."
                )
                for col in df.columns:
                    if col.startswith("sensor_") or col.startswith("op_setting_"):
                        logger.info(f"Converting sensor/op_setting column {col} to numeric")
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    elif col in [
                        "unit_number",
                        "time_in_cycles",
                        "unit",
                        "unit_id",
                        "engine_id",
                        "id",
                        "cycle",
                        "time",
                        "cycles",
                    ]:
                        logger.info(f"Converting identifier column {col} to Int64")
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

            self.debug_columns_at_each_step(df, "AFTER DATA TYPE VALIDATION")
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
            self.debug_columns_at_each_step(df, "BEFORE RANGE VALIDATION")

            logger.info("Validating data ranges and handling outliers...")

            # Get column specifications
            columns_spec = None
            if "columns" in self.config.schema:
                columns_spec = self.config.schema["columns"]
            elif "schema" in self.config.schema and "columns" in self.config.schema["schema"]:
                columns_spec = self.config.schema["schema"]["columns"]

            # Check if schema has column specifications (dictionary format)
            if isinstance(columns_spec, dict):
                for column, specs in columns_spec.items():
                    if (
                        column in df.columns
                        and isinstance(specs, dict)
                        and "range" in specs
                        and pd.api.types.is_numeric_dtype(df[column])
                    ):
                        min_val, max_val = specs["range"]

                        out_of_range_mask = (df[column] < min_val) | (df[column] > max_val)
                        num_outliers = out_of_range_mask.sum()

                        if num_outliers > 0:
                            logger.warning(
                                f"Found {num_outliers} outliers in column '{column}'. Capping values to range [{min_val}, {max_val}]."
                            )
                            df[column] = df[column].clip(lower=min_val, upper=max_val)
            else:
                logger.info("Schema columns is a list format. Skipping range validation.")

            self.debug_columns_at_each_step(df, "AFTER RANGE VALIDATION")
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
            self.debug_columns_at_each_step(df, "BEFORE MISSING VALUE HANDLING")

            logger.info("Handling missing values...")

            missing_before = df.isnull().sum()
            missing_cols_before = missing_before[missing_before > 0]
            if not missing_cols_before.empty:
                logger.info(f"Missing values before handling:\\n{missing_cols_before}")

            # Find unit identifier column
            unit_col = None
            possible_unit_names = ["unit_number", "unit", "unit_id", "engine_id", "id"]
            for col_name in possible_unit_names:
                if col_name in df.columns:
                    unit_col = col_name
                    logger.info(f"Using unit identifier column: {unit_col}")
                    break

            if unit_col is not None:
                logger.info(f"{unit_col} column dtype: {df[unit_col].dtype}")
                logger.info(f"{unit_col} has NaN values: {df[unit_col].isnull().any()}")

                # SIMPLE FIX: Skip groupby entirely to avoid losing the unit column
                logger.info("Using simple ffill/bfill to avoid groupby issues with unit column")
                df = df.ffill()
                df = df.bfill()
            else:
                logger.warning(
                    "Unit identifier column not found. Using simple ffill/bfill without grouping."
                )
                df = df.ffill()
                df = df.bfill()

            # For any remaining NaNs, fill with mean or mode
            for col in df.columns:
                if df[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fill_value = df[col].mean() if not df[col].isnull().all() else 0
                        df[col] = df[col].fillna(fill_value)
                        logger.warning(
                            f"Column '{col}' had remaining NaNs, filled with mean: {fill_value}."
                        )
                    elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(
                        df[col]
                    ):
                        fill_value = df[col].mode()[0] if not df[col].mode().empty else "unknown"
                        df[col] = df[col].fillna(fill_value)
                        logger.warning(
                            f"Column '{col}' had remaining NaNs, filled with mode: {fill_value}."
                        )

            self.debug_columns_at_each_step(df, "AFTER MISSING VALUE HANDLING")
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
            self.debug_columns_at_each_step(df, "BEFORE SAVING")

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
                logger.warning(
                    f"save_csv utility failed: {save_util_error}. Falling back to pandas to_csv."
                )
                df.to_csv(validated_file_path, index=False)
                logger.info(f"Successfully saved using pandas to_csv fallback.")

            if not validated_file_path.exists():
                raise IOError(f"File was not created: {validated_file_path}")

            file_size = validated_file_path.stat().st_size
            if file_size == 0:
                raise IOError(f"File was created but is empty: {validated_file_path}")

            # VERIFY THE SAVED FILE
            logger.info("VERIFYING SAVED FILE...")
            try:
                saved_df = pd.read_csv(validated_file_path)
                logger.info(f"Saved file shape: {saved_df.shape}")
                logger.info(f"Saved file columns: {list(saved_df.columns)}")

                if "unit_number" in saved_df.columns:
                    logger.info(
                        f"✓ unit_number successfully saved! Unique units: {saved_df['unit_number'].nunique()}"
                    )
                else:
                    logger.error("✗ unit_number NOT found in saved file!")

            except Exception as verify_error:
                logger.error(f"Error verifying saved file: {verify_error}")

            logger.info(
                f"Validated data successfully saved to {validated_file_path} (Size: {file_size} bytes)"
            )
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

            self.debug_columns_at_each_step(df, "RAW DATA LOADED")

            # 1. Validate columns
            if not self.validate_columns(df):
                logger.error("Data validation failed: Missing required columns.")
                raise ValueError("Missing required columns in raw data.")

            # CRITICAL CHANGE: DO NOT FILTER COLUMNS AT ALL
            # Comment out the column filtering logic completely
            logger.info("SKIPPING COLUMN FILTERING - PRESERVING ALL COLUMNS")

            # 2. Validate and correct data types
            df = self.validate_data_types(df)

            # 3. Validate ranges and handle outliers
            df = self.validate_ranges(df)

            # 4. Handle missing values
            df = self.handle_missing_values(df)

            # Final verification before saving
            if "unit_number" in df.columns:
                logger.info(
                    f"✓ FINAL CHECK: unit_number is present with {df['unit_number'].nunique()} unique units"
                )
            else:
                logger.error("✗ FINAL CHECK: unit_number is MISSING!")
                logger.error("Available columns at end of validation:", list(df.columns))

            # 5. Save validated data
            validated_file_path = self.save_validated_data(df, self.config.validated_data_path)

            logger.info(
                f"Data validation completed successfully. Final validated dataset contains {len(df)} records."
            )
            return validated_file_path

        except Exception as e:
            logger.error(f"An error occurred during the data validation pipeline: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise e
