import numpy as np
import pandas as pd
from binance.client import Client
from keras.layers import Conv1D
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas_ta as ta
from matplotlib import pyplot as plt
from config import Config


# ---------------------------
# ⚙️ Crypto Predictor Class
# ---------------------------
class CryptoPredictor:
    def __init__(self, symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1DAY, lookback="3650 days ago UTC",
                 seq_length=60, future_offset=5):
        """ Initialize CryptoPredictor """
        self.api_key = Config.BINANCE_API_KEY
        self.api_secret = Config.BINANCE_API_SECRET
        self.client = Client(self.api_key, self.api_secret)

        self.symbol = symbol
        self.interval = interval
        self.lookback = lookback
        self.seq_length = seq_length
        self.future_offset = future_offset

        self.df = None
        self.scaler = MinMaxScaler()
        self.model = None
        # For ensemble (if needed later)
        self.xgb_model = None

    # ---------------------------
    # 📊 Fetch Data
    # ---------------------------
    def fetch_data(self):
        """ Fetch historical data from Binance """
        try:
            klines = self.client.get_historical_klines(self.symbol, self.interval, self.lookback)
            self.df = pd.DataFrame(klines, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close_time', 'Quote_asset_volume', 'Number_of_trades',
                'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
            ])
            self.df['Date'] = pd.to_datetime(self.df['timestamp'], unit='ms')
            self.df.set_index('Date', inplace=True)
            float_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            self.df[float_cols] = self.df[float_cols].astype(float)
            print(f"✅ Data fetched: {len(self.df)} rows")
        except Exception as e:
            print(f"⚠️ Error fetching data: {e}")
            exit()

    # ---------------------------
    # 📈 Add Indicators
    # ---------------------------
    def add_indicators(self):
        """ Add technical indicators to enrich the feature set """
        if self.df is None or self.df.empty:
            print("⚠️ No data available. Fetch data first.")
            return

        # Basic moving averages
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['EMA_20'] = self.df['Close'].ewm(span=20).mean()
        # RSI using pandas_ta
        self.df['RSI_14'] = ta.rsi(self.df['Close'], length=14)
        # MACD (using histogram component)
        macd = ta.macd(self.df['Close'], fast=12, slow=26, signal=9)
        self.df['MACD'] = macd['MACDh_12_26_9']
        # Bollinger Bands (selecting lower, middle, and upper bands)
        bb = ta.bbands(self.df['Close'], length=20, std=2)
        self.df['BBL'] = bb['BBL_20_2.0']
        self.df['BBM'] = bb['BBM_20_2.0']
        self.df['BBU'] = bb['BBU_20_2.0']
        # Additional indicators for improved accuracy
        self.df['ADX'] = ta.adx(self.df['High'], self.df['Low'], self.df['Close'], length=14)['ADX_14']
        self.df['CCI'] = ta.cci(self.df['High'], self.df['Low'], self.df['Close'], length=20)
        self.df['OBV'] = ta.obv(self.df['Close'], self.df['Volume'])
        self.df.dropna(inplace=True)
        print("✅ Indicators added successfully!")

    # ---------------------------
    # 🔥 Sequence Generation with Future Target
    # ---------------------------
    def create_sequences(self):
        """ Create sequences from the scaled data with a future offset target """
        # Define the features to use
        features = ['Close', 'SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'BBL', 'BBM', 'BBU', 'ADX', 'CCI', 'OBV']
        valid_features = [col for col in features if col in self.df.columns]

        # Split data into training and testing portions
        train_size = int(0.8 * len(self.df))
        train_data = self.df[valid_features].iloc[:train_size]
        test_data = self.df[valid_features].iloc[train_size:]

        # Fit scaler on training data and transform both train and test data
        self.scaler.fit(train_data)
        train_scaled = self.scaler.transform(train_data)
        test_scaled = self.scaler.transform(test_data)

        # Define a helper function to create sequences
        def create(X, y, offset):
            sequences, targets = [], []
            for i in range(len(X) - self.seq_length - offset):
                seq = X[i:i + self.seq_length]
                # Future target: we predict the close price (first column) after offset
                target = y[i + self.seq_length + offset - 1, 0]
                sequences.append(seq)
                targets.append(target)
            return np.array(sequences), np.array(targets)

        X_train, y_train = create(train_scaled, train_scaled, self.future_offset)
        X_test, y_test = create(test_scaled, test_scaled, self.future_offset)

        print(f"✅ Sequences created: {X_train.shape}, {X_test.shape}")
        return X_train, X_test, y_train, y_test

    # ---------------------------
    # 🔥 Build LSTM-GRU Model with Attention
    # ---------------------------
    def build_model(self, input_shape):
        """ Simplified Model with CNN-LSTM """
        inputs = Input(shape=input_shape)

        # CNN for feature extraction
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)

        # LSTM for sequence learning
        x = LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)

        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)

        outputs = Dense(1)(x)

        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=AdamW(learning_rate=0.001), loss='mse')


        print("✅ Simplified CNN-LSTM model built successfully!")

    # ---------------------------
    # ✅ Train and Evaluate with Callbacks and Metrics
    # ---------------------------
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """ Train and evaluate the model with comprehensive metrics """
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_filename = f'model/model_{timestamp}.keras'
        model_checkpoint = ModelCheckpoint(model_filename, save_best_only=True)

        # Train the model using the callbacks and store training history
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,  # Adjust epochs as needed
            batch_size=32,  # Adjust batch size if necessary
            callbacks=[early_stopping, lr_scheduler, model_checkpoint],
            verbose=1
        )

        # Plot the training and validation loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # After training, make predictions on the test set
        y_pred = self.model.predict(X_test)

        # Reshape predictions if needed
        if len(y_pred.shape) == 3:
            y_pred = y_pred[:, -1, 0]
        else:
            y_pred = y_pred.reshape(-1)

        y_test = y_test.reshape(-1)

        # Inverse transform predictions and true values
        y_pred_inv = self.scaler.inverse_transform(
            np.hstack((y_pred.reshape(-1, 1), np.zeros((len(y_pred), X_test.shape[2] - 1))))
        )[:, 0]

        y_test_inv = self.scaler.inverse_transform(
            np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), X_test.shape[2] - 1))))
        )[:, 0]

        # Evaluate model performance
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)

        print("\n📊 Model Evaluation Metrics:")
        print(f"✅ MSE: {mse:.4f}")
        print(f"✅ RMSE: {rmse:.4f}")
        print(f"✅ MAE: {mae:.4f}")
        print(f"✅ R² Score: {r2:.4f}")

        # Plot actual vs predicted prices
        plt.figure(figsize=(14, 7))
        plt.style.use("seaborn-v0_8")
        plt.plot(y_test_inv, label="Actual Price", color="blue")
        plt.plot(y_pred_inv, label="Predicted Price", color="orange")
        plt.title("Actual vs Predicted Price")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()

# ---------------------------
# 🚀 Main Execution
# ---------------------------
if __name__ == "__main__":
    predictor = CryptoPredictor()
    predictor.fetch_data()
    predictor.add_indicators()
    X_train, X_test, y_train, y_test = predictor.create_sequences()

    if X_train is not None:
        predictor.build_model(X_train.shape[1:])
        predictor.train_and_evaluate(X_train, y_train, X_test, y_test)
    else:
        print("❌ No valid data. Exiting.")
