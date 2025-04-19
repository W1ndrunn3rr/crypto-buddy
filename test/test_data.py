from unittest import TestCase
from src.data_gatherers.data_collector import DataCollector, CryptoDataPoint
from src.data_gatherers.dataset import CryptoDataset
from dotenv import load_dotenv
import pandas as pd
import os
import numpy as np


class DataCollectorTest(TestCase):
    def setUp(self):
        load_dotenv()
        self.api_key = os.getenv("GECKO_API")
        self.dc = DataCollector(api_key=self.api_key)

    def test_fetch_crypto_data(self):
        data = self.dc.fetch_crypto_data(
            days=1, crypto_currency="bitcoin", currency="usd"
        )
        self.assertGreater(len(data), 0)

    def test_save_to_csv(self):
        data = [CryptoDataPoint(price=100, market_cap=200, total_volume=300)]
        self.dc.save_to_csv(data)
        self.assertTrue(os.path.exists("data_1.csv"))
        os.remove("data_1.csv")


class TestCryptoDataset(TestCase):
    def test_dataset_shapes(self):
        test_data = np.array(
            [
                [1.0, 10.0, 3.0],
                [2.0, 20.0, -2.9],
                [3.0, 30.0, 12.0],
                [4.0, 40.0, 3.0],
                [5.0, 50.0, 4.9],
            ]
        )
        window_size = 2

        dataset = CryptoDataset(test_data, window_size)

        assert len(dataset) == 3
        x, y = dataset[0]
        assert x.shape == (window_size, 3)
        assert y.shape == ()

    def test_mean_std(self):
        test_data = np.array(
            [
                [1.0, 10.0, 3.0],
                [2.0, 20.0, -2.9],
                [3.0, 30.0, 12.0],
                [4.0, 40.0, 3.0],
                [5.0, 50.0, 4.9],
            ]
        )
        window_size = 2

        dataset = CryptoDataset(test_data, window_size)
        mean = dataset.get_mean()
        std = dataset.get_std()

        assert mean.shape == (3,)
        assert std.shape == (3,)
        assert np.all(std != 0)
