"""
Download NASA C-MAPSS Turbofan Engine Degradation Dataset.

This script downloads the real NASA dataset used for predictive maintenance.
Dataset source: NASA Prognostics Data Repository
https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq

Note: This script expects real data to be available in the data/source directory.
If data is not present, it should be manually downloaded from the NASA repository.
"""

import os
import zipfile
from pathlib import Path
import urllib.request
import shutil

# Dataset URL (NASA C-MAPSS)
# Note: The dataset is available from NASA's data repository
DATASET_URL = "https://data.nasa.gov/download/xaut-bemq/application%2Fzip"
BACKUP_URL = (
    "https://raw.githubusercontent.com/makinarocks/"
    "awesome-industrial-machine-datasets/master/data-explanation/NASA/cmapss/CMAPSSData.zip"
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"
RAW_DIR = DATA_DIR / "raw"


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file from URL."""
    print(f"Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded to {dest_path}")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract zip file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def setup_directories() -> None:
    """Create necessary directories."""
    for dir_path in [DATA_DIR, SOURCE_DIR, RAW_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")


def check_existing_data() -> bool:
    """Check if real data already exists in the source directory."""
    required_files = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
    existing = all((SOURCE_DIR / f).exists() for f in required_files)
    if existing:
        print("Real data already exists in data/source directory")
    return existing


def download_cmapss_data() -> bool:
    """Download the NASA C-MAPSS dataset."""
    setup_directories()

    # Check if data already exists
    if check_existing_data():
        return True

    zip_path = DATA_DIR / "CMAPSSData.zip"

    # Try primary URL
    if not download_file(DATASET_URL, zip_path):
        # Try backup URL
        print("Trying backup URL...")
        if not download_file(BACKUP_URL, zip_path):
            print("\n" + "=" * 60)
            print("ERROR: Could not download dataset automatically.")
            print("Please manually download the NASA C-MAPSS dataset from:")
            print("https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data")
            print("\nPlace the following files in data/source/:")
            print("  - train_FD001.txt")
            print("  - test_FD001.txt")
            print("  - RUL_FD001.txt")
            print("=" * 60 + "\n")
            return False

    # Extract
    try:
        extract_zip(zip_path, SOURCE_DIR)

        # Move files if they're in a subdirectory
        for txt_file in SOURCE_DIR.rglob("*.txt"):
            if txt_file.parent != SOURCE_DIR:
                shutil.move(str(txt_file), SOURCE_DIR / txt_file.name)

        # Clean up zip
        zip_path.unlink()

        print("Dataset downloaded and extracted successfully!")
        return True

    except Exception as e:
        print(f"Error extracting: {e}")
        return False


def convert_to_csv() -> None:
    """Convert .txt files to CSV format for easier processing."""
    import pandas as pd

    columns = (
        ["unit_number", "time_in_cycles"]
        + [f"op_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    # Check if source files exist
    train_file = SOURCE_DIR / "train_FD001.txt"
    test_file = SOURCE_DIR / "test_FD001.txt"
    rul_file = SOURCE_DIR / "RUL_FD001.txt"

    if not all(f.exists() for f in [train_file, test_file, rul_file]):
        print("Source files not found in data/source/. Cannot convert to CSV.")
        print("Please ensure the following files exist:")
        print("  - data/source/train_FD001.txt")
        print("  - data/source/test_FD001.txt")
        print("  - data/source/RUL_FD001.txt")
        return

    # Convert train data
    print("Converting training data...")
    train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=columns)
    train_df.to_csv(RAW_DIR / "train.csv", index=False)
    print(f"  Saved {len(train_df)} training samples")

    # Convert test data
    print("Converting test data...")
    test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=columns)
    test_df.to_csv(RAW_DIR / "test.csv", index=False)
    print(f"  Saved {len(test_df)} test samples")

    # Convert RUL data
    print("Converting RUL data...")
    rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["RUL"])
    rul_df.to_csv(RAW_DIR / "rul.csv", index=False)
    print(f"  Saved {len(rul_df)} RUL values")


def main():
    """Main function to download and setup data."""
    print("=" * 60)
    print("NASA C-MAPSS Dataset Setup Script")
    print("=" * 60)

    # Download or verify dataset
    success = download_cmapss_data()

    if not success:
        print("\nSetup incomplete. Please provide the real dataset files.")
        return

    # Convert to CSV
    convert_to_csv()

    print("=" * 60)
    print("Dataset setup complete!")
    print("=" * 60)

    # Print data statistics
    if (RAW_DIR / "train.csv").exists():
        import pandas as pd

        train_df = pd.read_csv(RAW_DIR / "train.csv")
        test_df = pd.read_csv(RAW_DIR / "test.csv")

        print("\nDataset Statistics:")
        print(f"  Training samples: {len(train_df):,}")
        print(f"  Training units: {train_df['unit_number'].nunique()}")
        print(f"  Test samples: {len(test_df):,}")
        print(f"  Test units: {test_df['unit_number'].nunique()}")
        print(f"  Features: {len(train_df.columns)}")


if __name__ == "__main__":
    main()
