import os
import pickle
import json
import yaml
import pandas as pd
from typing import Any, Dict
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_yaml(file_path: str) -> Dict:
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading YAML file {file_path}: {e}")
        raise e
    
def save_yaml(data: Dict, file_path:str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.dump(data, file)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving YAML file {file_path}: {e}")
        raise e
    
def load_pickle(file_path:str) -> Any:
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {e}")
        raise e
    
def save_pickle(obj: Any, file_path: str):
    try:
        with open(file_path, 'wb') as file:
            return pickle.dump(obj, file)
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {e}")
        raise e
    

def load_csv(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}")
        raise e

def save_csv(df: pd.DataFrame, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"DataFrame saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving CSV file {file_path}: {e}")
        raise e