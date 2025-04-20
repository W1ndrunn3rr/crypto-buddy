from pydantic import BaseModel
from typing import List
import requests
import pandas as pd


class CryptoDataPoint(BaseModel):
    price: float
    market_cap: float
    total_volume: float


class CryptoDataResponse(BaseModel):
    prices: List[List[float]]
    market_caps: List[List[float]]
    total_volumes: List[List[float]]


class DataCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_crypto_data(
        self, days: int = 30, crypto_currency: str = "bitcoin", currency: str = "usd"
    ) -> List[CryptoDataPoint]:
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_currency}/market_chart?vs_currency={currency}&days={days}&interval=daily&precision=2&x_cg_demo_api_key={self.api_key}"
        response = requests.get(url)
        response.raise_for_status()

        data = CryptoDataResponse(**response.json())

        data_points = []
        for price, mcap, volume in zip(
            data.prices, data.market_caps, data.total_volumes
        ):
            data_points.append(
                CryptoDataPoint(
                    price=price[1], market_cap=mcap[1], total_volume=volume[1]
                )
            )

        return data_points

    def save_to_csv(self, data: List[CryptoDataPoint], path: str) -> None:
        data_dicts = [item.model_dump() for item in data]

        df = pd.DataFrame(data_dicts)
        df = df[["price", "market_cap", "total_volume"]]
        df.to_csv(path, index=False)

        print(f"Data saved to {path}")
