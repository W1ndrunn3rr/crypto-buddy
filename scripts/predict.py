from src.model.lstm import MultiStepLSTM
from src.data_gatherers.data_collector import DataCollector
import torch
from dotenv import load_dotenv
import os


def predict(currency: str = "bitcoin", forecast_days: int = 7):
    load_dotenv(".env")
    model = MultiStepLSTM(input_size=3, hidden_size=128)
    model.load_state_dict(torch.load(f"data/{currency}_model.pth"))

    stats = torch.load(f"data/{currency}_normalization_stats.pt", weights_only=False)
    std, mean = stats["std"], stats["mean"]

    dc = DataCollector(api_key=os.getenv("GECKO_API"))

    response = dc.fetch_crypto_data(int(forecast_days) - 1, currency, "usd")

    input_tensor: torch.tensor = torch.tensor(
        [[point.price, point.market_cap, point.total_volume] for point in response],
        dtype=torch.float32,
    )

    normalized_tensor = (
        ((input_tensor - mean) / std).clone().detach().unsqueeze(1).float()
    )

    with torch.no_grad():
        model.eval()
        prediction = model(normalized_tensor).squeeze()
        for i in range(len(prediction)):
            predicted_price = prediction[i] * std[0] + mean[0]
            print("────────────────────────────────────────────")
            print(f"CURRENCY: {currency.upper()}")
            print(f"PREDICTION FOR DAY {i+1}:")
            print("────────────────────────────────────────────")
            print(f"PREDICTION : {round(predicted_price.item(),2)} USD")
            print(f"STD: {std}")
            print(f"MEAN: {mean}")
            print("────────────────────────────────────────────")
