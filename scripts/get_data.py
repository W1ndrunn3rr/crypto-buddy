from dotenv import load_dotenv
import os
from src.data_gatherers.data_collector import DataCollector


def get_data(currency: str = "bitcoin"):
    load_dotenv()
    api_key = os.getenv("GECKO_API")
    dc = DataCollector(api_key)

    res = dc.fetch_crypto_data(359, currency, "usd")

    dc.save_to_csv(res, f"data/{currency}_data.csv")
    print(f"Data for {currency} saved to data/{currency}_data.csv")
