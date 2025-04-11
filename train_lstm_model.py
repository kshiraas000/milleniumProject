import os
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime

def train_and_save_model(symbol):
    # year of historical data and storing in dataframe
    stock = yf.Ticker(symbol)
    df = stock.history(period="1y")

    if df.empty or len(df) < 100:
        raise Exception("Not enough data to train.")

    # adding technical indicators
    df['SMA_10'] = ta.sma(df['Close'], length=10)
    df['EMA_10'] = ta.ema(df['Close'], length=10)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df = df.dropna()

    # normalize features
    feature_cols = ['Close', 'Volume', 'SMA_10', 'EMA_10', 'RSI']
    data = df[feature_cols].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # create sequences for LSTM
    # x = 60 day sliding window of input data
    # y = next 7 days of closing prices
    X, y = [], []
    window_size = 60
    for i in range(window_size, len(scaled_data) - 7):
        X.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i:i+7, 0])  # Close only

    X, y = np.array(X), np.array(y)

    # building and training LSTM
    # using mean squared error and adam optimizer
    # MSE = loss function to evaluate how well model's predictions match actual values
    # average squared difference between predicted and true values (closer to 0 the better)
    # Adam is optimization algorithm that updates model wieghts during training
    # so tweaks models to reudce MSE or error (handles noisy and fast convergence)
    # combines momentum and adaptive learnign rates
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(7))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    # Save model and scaler
    os.makedirs("models", exist_ok=True)
    model.save("models/aapl_lstm.h5", include_optimizer=False)
    np.save(f"models/{symbol.lower()}_scaler.npy", scaler.fit(data))  # Save scaler to retrain easily
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    model.save(f"models/{symbol.lower()}_lstm_{timestamp}.h5", include_optimizer=False)

    print(f"📊 Training model for {symbol} on data from {df.index.min().date()} to {df.index.max().date()}")

if __name__ == "__main__":
    train_and_save_model("AAPL")
