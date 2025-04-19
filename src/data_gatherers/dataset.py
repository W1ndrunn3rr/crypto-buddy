from torch.utils.data import Dataset
import numpy as np
import torch
from typing import Tuple
from sklearn.preprocessing import StandardScaler


class CryptoDataset(Dataset):
    def __init__(self, data: np.ndarray, window_size: int = 3):
        self.mean, self.std = data.mean(axis=0), data.std(axis=0)
        self.std = np.where(self.std == 0, 1, self.std)
        self.scaler = StandardScaler()

        data = self.scaler.fit_transform(data)

        total_samples = len(data) - window_size

        X_np = np.zeros((total_samples, window_size, data.shape[1]))
        y_np = np.zeros(total_samples)

        for i in range(total_samples):
            X_np[i] = data[i : i + window_size]
            y_np[i] = data[i + window_size, 0]

        self.X = torch.tensor(X_np, dtype=torch.float32)
        self.y = torch.tensor(y_np, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def get_mean(self) -> np.ndarray:
        return self.mean

    def get_std(self) -> np.ndarray:
        return self.std
