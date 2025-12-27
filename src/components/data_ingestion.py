import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


MONGO_URL = os.getenv("MONGO_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

client = MongoClient(MONGO_URL)
db = client[DATABASE_NAME]
collection = client[COLLECTION_NAME]

files = {
    "train_data": "data/raw/train.csv",
    "test_data": "data/raw/test.csv",
    "rul_data": "data/raw/rul.csv",
}

for collection_name, file_path in files.items():
    print(f"📥 Ingesting {file_path} into collection `{collection_name}`...")
    df = pd.read_csv(file_path)
    data = df.to_dict(orient="records")
    db[collection_name].delete_many({})  # Optional: Clear old data
    db[collection_name].insert_many(data)
    print(f"✅ Inserted {len(data)} records into `{collection_name}`.")

print("Data ingestion successful")
