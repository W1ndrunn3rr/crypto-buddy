import pandas as pd
import os
import torch
from src.data_gatherers.dataset import CryptoDataset
from src.model.lstm import MultiStepLSTM
import os


def train():
    os.system("clear")
    f_data = pd.read_csv("data/bitcoin_data.csv")

    train_data = f_data.iloc[: int(len(f_data) * 0.8)]
    test_data = f_data.iloc[int(len(f_data) * 0.8) :]

    train_dataset = CryptoDataset(train_data.values, window_size=30)
    test_dataset = CryptoDataset(test_data.values, window_size=30)

    train_mean = train_dataset.get_mean()
    train_std = train_dataset.get_std()

    torch.save(
        {
            "mean": train_mean,
            "std": train_std,
        },
        "data/normalization_stats.pt",
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiStepLSTM(input_size=3, hidden_size=128).to(device)

    optimizer = torch.optim.RAdam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-5, max_lr=0.01, step_size_up=50, cycle_momentum=False
    )

    criterion = torch.nn.HuberLoss(delta=0.3)

    epochs = 100

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            y_pred = model(X, forecast_size=1).squeeze()
            loss = criterion(y_pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            grad_norms = [
                p.grad.abs().mean().item()
                for p in model.parameters()
                if p.grad is not None
            ]
            # print(
            #     f"Gradient norms - Min: {min(grad_norms):.2e} | Max: {max(grad_norms):.2e}"
            # )

        scheduler.step()
        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)

                y_pred = model(X, forecast_size=1).squeeze()
                loss = criterion(y_pred, y)
                val_loss += loss.item()

                # print(f"Predictions range: {y_pred.min():.3f} - {y_pred.max():.3f}")
                # print(f"True range: {y_pred.min():.3f} - {y_pred.max():.3f}")

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"[{epoch:03d}/{epochs}] ", end="")
            print(f"Train: {train_loss/len(train_loader):.4f} ", end="")
            print(f"Val: {val_loss/len(test_loader):.4f}", end="")
            print(" |" + "█" * int(50 * epoch / epochs) + f" {100*epoch/epochs:.0f}%")

    print("Training complete")
    print("Saving model...")
    os.makedirs("data", exist_ok=True)
    torch.save(model.state_dict(), "data/model.pth")
