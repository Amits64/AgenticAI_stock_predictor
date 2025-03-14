import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from binance.client import Client
import time
from datetime import datetime, timedelta

# Set up Binance API client
api_key = 'FpmgOpAE2bez7ct136mQVPdRt6lbanMnuDK54iqP0l928bQ13pAN5VPKuqH71XK4'
api_secret = 'KWmXhMKxAvkofRbrsOrLusKLB351t6kBBAKzHDlOFd53y2uNnX88vtj73czZls3j'
client = Client(api_key, api_secret)

# Define the symbol and the time frame for historical data
symbol = "BTCUSDT"  # Bitcoin to USD trading pair
interval = Client.KLINE_INTERVAL_1DAY  # 1-day candlesticks
lookback_period = "3650 days ago UTC"  # 3650 days of historical data

# Fetch historical data from Binance
try:
    klines = client.get_historical_klines(symbol, interval, lookback_period)

    # Convert data into a DataFrame
    df = pd.DataFrame(klines, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time',
                                       'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume',
                                       'Taker_buy_quote_asset_volume', 'Ignore'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)

    # Keep only relevant columns (Close price for prediction)
    df = df[['Close']]
    df = df.dropna()  # Drop any NaN values

except Exception as e:
    print(f"Error fetching data from Binance: {e}")
    exit()

# Normalize the data
scaler = MinMaxScaler()
df['Close'] = scaler.fit_transform(df[['Close']])

# Save the scaler for later use
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Add technical indicators to the dataset
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

window = 20
std_dev = df['Close'].rolling(window=window).std()
df['BB_upper'] = df['EMA_20'] + (2 * std_dev)
df['BB_lower'] = df['EMA_20'] - (2 * std_dev)

df = df.dropna()

# Normalize all columns again after technical indicator addition
df[['Close', 'SMA_20', 'EMA_20', 'RSI', 'BB_upper', 'BB_lower']] = scaler.fit_transform(
    df[['Close', 'SMA_20', 'EMA_20', 'RSI', 'BB_upper', 'BB_lower']]
)


# Create sequences for LSTM
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


SEQ_LENGTH = 50
X, y = create_sequences(df[['Close', 'SMA_20', 'EMA_20', 'RSI', 'BB_upper', 'BB_lower']].values, SEQ_LENGTH)

X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

# Split into training and test datasets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the LSTM model with gradient clipping
model = Sequential([
    Bidirectional(LSTM(200, activation='relu', return_sequences=True, input_shape=(SEQ_LENGTH, X_train.shape[2]))),
    Dropout(0.3),
    LSTM(200, activation='relu', return_sequences=True),
    Dropout(0.3),
    GRU(100, activation='relu', return_sequences=False),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001, clipvalue=5.0), loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test),
                    shuffle=False, callbacks=[early_stopping])

# Save the model
model.save("ai_model.keras")
print("✅ Model saved as ai_model.keras")
