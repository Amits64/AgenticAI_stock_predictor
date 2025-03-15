import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from binance.client import Client
from technical_analysis import add_technical_indicators  # Import the function

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

    # Keep only relevant columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.dropna()  # Drop any NaN values

    # Verify 'Volume' column exists
    if 'Volume' not in df.columns:
        raise KeyError("The 'Volume' column is missing from the DataFrame.")

except Exception as e:
    print(f"Error fetching data from Binance: {e}")
    exit()

# Add technical indicators to the dataset
df = add_technical_indicators(df)
df = df.dropna()

# Normalize the data
scaler = MinMaxScaler()
columns_to_normalize = ['Close', 'Open', 'High', 'Low', 'SMA_20', 'EMA_20', 'RSI',
                        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
                        'ICH_SenkouSpanA_9_26_52', 'ICH_SenkouSpanB_9_26_52', 'Supertrend', 'VWAP', 'ATR', 'Volatility']
if 'Volume' in df.columns:
    columns_to_normalize.append('Volume')

# Ensure all columns to normalize exist in the DataFrame
columns_to_normalize = [col for col in columns_to_normalize if col in df.columns]

df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Save the scaler for later use
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Create sequences for LSTM
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predicting the 'Close' price
    return np.array(X), np.array(y)

SEQ_LENGTH = 50
X, y = create_sequences(df.values, SEQ_LENGTH)

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