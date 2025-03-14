import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler  # Changed to MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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

# Normalize the data (ensure no NaN or Inf values are in the input data)
scaler = MinMaxScaler()  # Changed to MinMaxScaler
df['price'] = scaler.fit_transform(df[['price']])

# Save the scaler for later use
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Check for NaN or Inf values in the dataset
if np.any(np.isnan(df['price'])) or np.any(np.isinf(df['price'])):
    print("Data contains NaN or Inf values. Please clean the data.")
    exit()

# Add technical indicators to the dataset (SMA_20, EMA_20, RSI, etc.)
# Note: You can modify this part to add more technical indicators as needed.

# Simple Moving Average (SMA_20)
df['SMA_20'] = df['price'].rolling(window=20).mean()

# Exponential Moving Average (EMA_20)
df['EMA_20'] = df['price'].ewm(span=20, adjust=False).mean()

# Relative Strength Index (RSI)
delta = df['price'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Bollinger Bands (BB)
window = 20
std_dev = df['price'].rolling(window=window).std()
df['BB_upper'] = df['EMA_20'] + (2 * std_dev)
df['BB_lower'] = df['EMA_20'] - (2 * std_dev)

# Ensure no NaN values after adding technical indicators
df = df.dropna()

# Normalize all features (including technical indicators)
df[['price', 'SMA_20', 'EMA_20', 'RSI', 'BB_upper', 'BB_lower']] = scaler.fit_transform(
    df[['price', 'SMA_20', 'EMA_20', 'RSI', 'BB_upper', 'BB_lower']]
)

# Save the updated scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Check for NaN or Inf values again after adding indicators
if np.any(np.isnan(df[['price', 'SMA_20', 'EMA_20', 'RSI', 'BB_upper', 'BB_lower']].values)) or \
        np.any(np.isinf(df[['price', 'SMA_20', 'EMA_20', 'RSI', 'BB_upper', 'BB_lower']].values)):
    print("Data contains NaN or Inf values after adding indicators. Please clean the data.")
    exit()

# Create sequences for LSTM
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 50
X, y = create_sequences(df[['price', 'SMA_20', 'EMA_20', 'RSI', 'BB_upper', 'BB_lower']].values, SEQ_LENGTH)

# Reshape X to 3D for LSTM input (samples, time steps, features)
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

# Compile the model with gradient clipping
model.compile(optimizer=Adam(learning_rate=0.001, clipvalue=5.0), loss='mse')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test),
                    shuffle=False, callbacks=[early_stopping])

# Save the model
model.save("ai_model.keras")
print("✅ Model saved as ai_model.keras")
