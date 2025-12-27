import os
from datetime import datetime

# File paths
CONFIG_FILE_PATH = "config/config.yaml"
SCHEMA_FILE_PATH = "config/schema.yaml"

# Data paths
RAW_DATA_DIR = "data/raw/train.csv"
VALIDATED_DATA_DIR = "data/validated"
TRANSFORMED_DATA_DIR = "data/transformed"
PREDICTIONS_DIR = "data/predictions"

# Artifacts paths
MODEL_DIR = "artifacts/models"
LOGS_DIR = "artifacts/logs"

# Model names
LSTM_MODEL_NAME = "lstm_model.h5"
SCALER_NAME = "scaler.pkl"

# Training configuration
TRAIN_PIPELINE_NAME = "training_pipeline"
BATCH_PIPELINE_NAME = "batch_pipeline"
