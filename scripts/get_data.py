from dotenv import load_dotenv
import os
from src.data_gatherers.data_collector import DataCollector


def get_data():
    load_dotenv()
    api_key = os.getenv("GECKO_API")
    dc = DataCollector(api_key)

    res = dc.fetch_crypto_data(359, "ethereum", "usd")

    dc.save_to_csv(res)
