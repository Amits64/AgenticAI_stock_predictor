# train_model.py
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from pycoingecko import CoinGeckoAPI
import time
from datetime import datetime, timedelta

# Initialize CoinGecko API client
cg = CoinGeckoAPI()

# Get the current timestamp
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Convert to Unix timestamp
start_timestamp = int(time.mktime(start_date.timetuple()))
end_timestamp = int(time.mktime(end_date.timetuple()))

# Fetch historical market data for the last 365 days
ticker = "bitcoin"  # Use the CoinGecko ID for Bitcoin (not the symbol)
try:
    data = cg.get_coin_market_chart_range_by_id(id=ticker, vs_currency='usd', from_timestamp=start_timestamp,
                                                to_timestamp=end_timestamp)

    # Convert the data into a DataFrame
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert timestamp to Date
    df.set_index('Date', inplace=True)
    df = df[['price']]  # Keep only the 'price' column
    df = df.dropna()  # Drop any NaN values
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
df['price'] = scaler.fit_transform(df[['price']])

# Save the scaler for later use
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


# Create sequences for LSTM
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


SEQ_LENGTH = 50
X, y = create_sequences(df['price'].values, SEQ_LENGTH)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Define LSTM Model
model = Sequential([
    LSTM(100, activation='relu', return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    LSTM(100, activation='relu'),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(1)
])

# Compile & Train Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, shuffle=False)

# Save the Model in `.keras` format
model.save("ai_model.keras")
print("✅ Model saved as ai_model.keras")
