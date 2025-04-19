import torch
from torch import nn


class MultiStepLSTM(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.linear_block = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
        )

        self.output = nn.Linear(hidden_size, 1)

        for name, param in self.named_parameters():
            if "weight" in name and "lstm" in name:
                nn.init.orthogonal_(param)

    def forward(self, x: torch.Tensor, forecast_size: int = 1) -> torch.Tensor:

        x, _ = self.lstm(x)

        x = x[:, -1, :]

        x = self.linear_block(x)

        x = self.output(x)

        return x[:forecast_size, :]
