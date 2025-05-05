import os
import pickle
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# -----------------------------
# USER CONFIGURATION
# -----------------------------
TICKERS = ["NVDA", "TSLA", "JPM", "AMD"]

# -----------------------------
# SYSTEM CONFIGURATION
# -----------------------------
ARIMA_ORDER = (5, 1, 0)
FORECAST_HORIZON = 10

MODEL_DIR = "arima_models"
os.makedirs(MODEL_DIR, exist_ok=True)
print("Models will be saved to:", MODEL_DIR)

# -----------------------------
# FUNCTIONS
# -----------------------------


def train_and_save_arima(ticker, order, model_dir):
    print(f"\n⏳ Training ARIMA{order} for {ticker}...")
    df = yf.download(ticker, start="2010-01-01",
                     progress=False, auto_adjust=True)
    ts = df["Close"].asfreq("B").ffill()
    model = ARIMA(ts, order=order)
    fit = model.fit()
    out_path = os.path.join(model_dir, f"{ticker}_arima.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(fit, f)
    print(f"✅ Saved model to {out_path}")


def load_and_forecast(ticker, model_dir, steps):
    model_path = os.path.join(model_dir, f"{ticker}_arima.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model for {ticker} at {model_path}")
    with open(model_path, "rb") as f:
        fit = pickle.load(f)
    return fit.forecast(steps=steps)


# -----------------------------
# BATCH TRAIN
# -----------------------------
for tk in TICKERS:
    train_and_save_arima(tk, ARIMA_ORDER, MODEL_DIR)

print("\n🏁 Batch training complete!")

# -----------------------------
# EXAMPLE FORECAST
# -----------------------------
example = TICKERS[0]
print(f"\n📈 Next {FORECAST_HORIZON} business days for {example}:")
print(load_and_forecast(example, MODEL_DIR, FORECAST_HORIZON))
