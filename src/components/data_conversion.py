import logging
import os
from pathlib import Path

import pandas as pd

from src.utils.file_ops import create_directories, read_yaml

logging.basicConfig(level=logging.INFO)


class DataIngestion:
    def __init__(self, config_path: str, schema_path: str):
        self.config = read_yaml(config_path)
        self.schema = read_yaml(schema_path)

        # Paths
        self.raw_data_path = Path(self.config["data"]["raw_data_path"])
        self.train_file = Path(self.config["data"]["train_file"])
        self.test_file = Path(self.config["data"]["test_file"])
        self.rul_file = Path(self.config["data"]["rul_file"])

        self.column_names = self.schema["columns"]

    def load_txt_as_df(self, file_path: Path) -> pd.DataFrame:
        """
        Load a .txt file into a DataFrame, remove empty cols, assign schema headers.
        """
        df = pd.read_csv(file_path, sep=r"\s+", header=None)
        df = df.dropna(axis=1, how="all")
        df.columns = self.column_names
        return df

    def run(self):
        """
        Run data ingestion pipeline.
        """
        create_directories([self.raw_data_path])

        logging.info(f"🔹 Reading: {self.train_file}")
        train_df = self.load_txt_as_df(self.train_file)

        logging.info(f"🔹 Reading: {self.test_file}")
        test_df = self.load_txt_as_df(self.test_file)

        logging.info(f"🔹 Reading: {self.rul_file}")
        rul_df = pd.read_csv(self.rul_file, sep=r"\s+", header=None).dropna(axis=1, how="all")
        rul_df.columns = ["RUL"]

        train_df.to_csv(self.raw_data_path / "train.csv", index=False)
        test_df.to_csv(self.raw_data_path / "test.csv", index=False)
        rul_df.to_csv(self.raw_data_path / "rul.csv", index=False)

        logging.info("✅ Data ingestion completed successfully.")
        return train_df, test_df, rul_df
