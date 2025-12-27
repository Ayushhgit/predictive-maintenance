import os
import pathlib

import yaml


def read_yaml(path_to_yaml: str) -> dict:
    """
    Reads a YAML file and returns the content as a Python dictionary.
    Args:
        path_to_yaml (str): Path to the YAML file.
    Returns:
        dict: Parsed YAML content.
    """
    try:
        with open(path_to_yaml, "r") as file:
            content = yaml.safe_load(file)
            return content
    except Exception as e:
        raise Exception(f"Error reading the YAML file at {path_to_yaml}: {e}")


def create_directories(paths: list):
    """
    Creates directories from a list of paths if they do not exist.
    Args:
        paths (list): List of directory paths to create.
    """
    for path in paths:
        dir_path = pathlib.Path(path)
        os.makedirs(dir_path, exist_ok=True)
