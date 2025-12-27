import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient


def load_data_as_df(db_name, collection_name):
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URL")
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    data = list(collection.find({}, {"_id": False}))
    return pd.DataFrame(data)
