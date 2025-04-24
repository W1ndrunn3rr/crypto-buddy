# Cryptocurrency Price Prediction with LSTM

This project utilizes an LSTM (Long Short-Term Memory) neural network to predict cryptocurrency prices based on historical data. It fetches cryptocurrency data using the CoinGecko API and provides a streamlined pipeline for training and prediction.

## Features

- **LSTM Model**: Deep learning model for time-series forecasting of cryptocurrency prices.
- **CoinGecko API Integration**: Fetches up-to-date or historical cryptocurrency data.
- **Poetry**: Used for dependency management and version control.
- **Modular Pipeline**: Supports data fetching, model training, and prediction.
- **Libraries**: Built with PyTorch, NumPy, scikit-learn, and pandas.

## Prerequisites

- Python 3.8+
- Poetry (for dependency management)
- CoinGecko API key (for fetching new data)

## Installation
1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-directory>
```
2. Install dependencies using Poetry:
```bash
poetry install
 ```

## Configuration

Set your CoinGecko API key as an environment variable:
```bash
export GECKO_API="your_api_key_here"
```

## Usage

### Fetching Data
To download cryptocurrency data (e.g., Bitcoin):
```bash
poetry run getdata bitcoin
```

### Training the Model
To train the model on existing data:
```bash
poetry run train bitcoin
```
*Note:* If data doesn't exist, run `getdata` first.

### Making Predictions
To generate predictions for currency (e.g., Bitcoin) for 3 days (default: 7-day forecast):
```bash
poetry run predict bitcoin 3
```

### Full Pipeline
To execute the entire pipeline (fetch data → train → predict) [for example bitcoin for 21 days forecast]:
```bash
chmod +x run.sh
./run.sh bitcoin 21
```

## Project Structure
```
.
├── data/               # Stores cryptocurrency data
├── models/             # Trained model checkpoints
├── src/                # Source code
│   ├── data_loader.py  # Data fetching and preprocessing
│   ├── model.py        # LSTM implementation
│   ├── train.py        # Training script
│   └── predict.py      # Prediction script
├── pyproject.toml      # Poetry configuration
└── run.sh              # Full pipeline script
```

## Notes
- For training, pre-downloaded data can be used.
- Predictions require fresh API calls due to market volatility.
- Default prediction horizon is 7 days (configurable in code).
- Project uses USD currency 


## Important disclaimer
This is a **HOBBY PROJECT** for educational purposes only.  
- The Software **must not be used as a predictor for real cryptocurrency markets**.  
- The author **bears no responsibility** for any financial losses, damages,  
  or decisions made based on this Software.  
- **Do not use this Software for profit-driven activities**
MIT License

Copyright (c) 2025 [Your Name or Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

---
**IMPORTANT DISCLAIMER**:  
This is a **HOBBY PROJECT** for educational purposes only.  
- The Software **must not be used as a predictor for real cryptocurrency markets**  
- The author **bears no responsibility** for any financial losses, damages,  
  or decisions made based on this Software  
- **Do not use this Software for profit-driven activities**  
- All cryptocurrency data used by this Software is property of **© 2025 CoinGecko** and is subject to CoinGecko's Terms of Service (https://www.coingecko.com/en/terms)
