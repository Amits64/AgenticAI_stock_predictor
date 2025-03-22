import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization,
                                     Bidirectional, LSTM, Dense, Dropout,
                                     Attention, Concatenate)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas_ta as ta
from matplotlib import pyplot as plt
from config import Config
import keras_tuner as kt

# ---------------------------
# ⚙️ Advanced Crypto Predictor Class
# ---------------------------
class CryptoPredictor:
    def __init__(self, symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1DAY, lookback="3650 days ago UTC",
                 seq_length=60, future_offset=5):
        """Initialize predictor and set parameters."""
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
        self.model_a = None
        self.model_b = None
        self.input_shape = None

    # ---------------------------
    # 📊 Fetch Data
    # ---------------------------
    def fetch_data(self):
        """Fetch historical data from Binance."""
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
    # 📈 Add Technical Indicators
    # ---------------------------
    def add_indicators(self):
        """Add technical indicators to enrich the feature set."""
        if self.df is None or self.df.empty:
            print("⚠️ No data available. Fetch data first.")
            return

        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['EMA_20'] = self.df['Close'].ewm(span=20).mean()
        self.df['RSI_14'] = ta.rsi(self.df['Close'], length=14)
        macd = ta.macd(self.df['Close'], fast=12, slow=26, signal=9)
        self.df['MACD'] = macd['MACDh_12_26_9']
        bb = ta.bbands(self.df['Close'], length=20, std=2)
        self.df['BBL'] = bb['BBL_20_2.0']
        self.df['BBM'] = bb['BBM_20_2.0']
        self.df['BBU'] = bb['BBU_20_2.0']
        self.df['ADX'] = ta.adx(self.df['High'], self.df['Low'], self.df['Close'], length=14)['ADX_14']
        self.df['CCI'] = ta.cci(self.df['High'], self.df['Low'], self.df['Close'], length=20)
        self.df['OBV'] = ta.obv(self.df['Close'], self.df['Volume'])
        self.df.dropna(inplace=True)
        print("✅ Indicators added successfully!")

    # ---------------------------
    # 🔥 Create Sequences for Training
    # ---------------------------
    def create_sequences(self):
        """Create sequences from the scaled data with a future target."""
        features = ['Close', 'SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'BBL', 'BBM', 'BBU', 'ADX', 'CCI', 'OBV']
        valid_features = [col for col in features if col in self.df.columns]

        train_size = int(0.8 * len(self.df))
        train_data = self.df[valid_features].iloc[:train_size]
        test_data = self.df[valid_features].iloc[train_size:]

        self.scaler.fit(train_data)
        train_scaled = self.scaler.transform(train_data)
        test_scaled = self.scaler.transform(test_data)

        def create(X, y, offset):
            sequences, targets = [], []
            for i in range(len(X) - self.seq_length - offset):
                seq = X[i:i + self.seq_length]
                target = y[i + self.seq_length + offset - 1, 0]
                sequences.append(seq)
                targets.append(target)
            return np.array(sequences), np.array(targets)

        X_train, y_train = create(train_scaled, train_scaled, self.future_offset)
        X_test, y_test = create(test_scaled, test_scaled, self.future_offset)
        print(f"✅ Sequences created: {X_train.shape}, {X_test.shape}")
        return X_train, X_test, y_train, y_test

    # ---------------------------
    # 🔥 Define Model Architectures with Hyperparameter Search
    # ---------------------------
    def build_model_a(self, hp):
        """Advanced model: CNN + Bidirectional LSTM + Attention."""
        inputs = Input(shape=self.input_shape)
        cnn_filters = hp.Int("cnn_filters", 16, 64, step=16, default=32)
        x = Conv1D(filters=cnn_filters, kernel_size=hp.Choice("kernel_size", [3, 5]), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        lstm_units = hp.Int("lstm_units", 32, 128, step=32, default=64)
        x = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(0.01)))(x)
        x = Dropout(hp.Float("dropout_lstm", 0.2, 0.5, step=0.1, default=0.3))(x)
        attention = Attention()([x, x])
        attention = BatchNormalization()(attention)
        combined = Concatenate()([x, attention])
        lstm_units2 = hp.Int("lstm_units2", 16, 64, step=16, default=32)
        x = LSTM(lstm_units2, return_sequences=False)(combined)
        x = Dropout(hp.Float("dropout_dense", 0.2, 0.5, step=0.1, default=0.3))(x)
        dense_units = hp.Int("dense_units", 16, 64, step=16, default=32)
        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(hp.Float("dropout_dense2", 0.2, 0.5, step=0.1, default=0.3))(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        lr = hp.Float("lr", 1e-4, 1e-2, sampling="log", default=1e-3)
        model.compile(optimizer=AdamW(learning_rate=lr), loss='mse')
        return model

    def build_model_b(self, hp):
        """Simpler model: CNN + LSTM."""
        inputs = Input(shape=self.input_shape)
        cnn_filters = hp.Int("cnn_filters", 16, 64, step=16, default=32)
        x = Conv1D(filters=cnn_filters, kernel_size=hp.Choice("kernel_size", [3, 5]), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        lstm_units = hp.Int("lstm_units", 32, 128, step=32, default=64)
        x = LSTM(lstm_units, return_sequences=False, kernel_regularizer=l2(0.01))(x)
        x = Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1, default=0.3))(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        lr = hp.Float("lr", 1e-4, 1e-2, sampling="log", default=1e-3)
        model.compile(optimizer=AdamW(learning_rate=lr), loss='mse')
        return model

    # ---------------------------
    # 🔍 Hyperparameter Tuning Utility
    # ---------------------------
    def tune_model(self, build_fn, X_train, y_train, X_val, y_val, max_trials=10, executions_per_trial=1, project_name="tuner"):
        tuner = kt.RandomSearch(
            build_fn,
            objective="val_loss",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory="tuner_dir",
            project_name=project_name,
            overwrite=True
        )
        tuner.search(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=50,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                     verbose=1)
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("✅ Best hyperparameters for", project_name, ":", best_hp.values)
        return tuner.hypermodel.build(best_hp)

    # ---------------------------
    # ✅ Train, Tune, and Evaluate with Ensemble
    # ---------------------------
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        # Split training data for tuning
        split_idx = int(0.9 * len(X_train))
        X_tr, y_tr = X_train[:split_idx], y_train[:split_idx]
        X_val, y_val = X_train[split_idx:], y_train[split_idx:]
        self.input_shape = X_train.shape[1:]

        print("Tuning Model A (Advanced)...")
        model_a = self.tune_model(self.build_model_a, X_tr, y_tr, X_val, y_val, project_name="model_a")
        print("Tuning Model B (Simpler)...")
        model_b = self.tune_model(self.build_model_b, X_tr, y_tr, X_val, y_val, project_name="model_b")

        # Callbacks for full training
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_a = ModelCheckpoint(f'model/model_a_{timestamp}.keras', save_best_only=True)
        checkpoint_b = ModelCheckpoint(f'model/model_b_{timestamp}.keras', save_best_only=True)

        print("Training Model A on full training data...")
        history_a = model_a.fit(X_train, y_train,
                                validation_data=(X_test, y_test),
                                epochs=100,
                                batch_size=32,
                                callbacks=[early_stopping, lr_scheduler, checkpoint_a],
                                verbose=1)
        print("Training Model B on full training data...")
        history_b = model_b.fit(X_train, y_train,
                                validation_data=(X_test, y_test),
                                epochs=100,
                                batch_size=32,
                                callbacks=[early_stopping, lr_scheduler, checkpoint_b],
                                verbose=1)

        # Ensemble prediction: average predictions from both models
        print("Ensembling predictions...")
        y_pred_a = model_a.predict(X_test)
        y_pred_b = model_b.predict(X_test)
        y_pred_a = y_pred_a.reshape(-1)
        y_pred_b = y_pred_b.reshape(-1)
        ensemble_pred = (y_pred_a + y_pred_b) / 2.0
        y_test_flat = y_test.reshape(-1)

        # Inverse transform predictions (only the first column, assuming it is 'Close')
        y_pred_inv = self.scaler.inverse_transform(
            np.hstack((ensemble_pred.reshape(-1, 1), np.zeros((len(ensemble_pred), X_test.shape[2] - 1))))
        )[:, 0]
        y_test_inv = self.scaler.inverse_transform(
            np.hstack((y_test_flat.reshape(-1, 1), np.zeros((len(y_test_flat), X_test.shape[2] - 1))))
        )[:, 0]

        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)

        print("\n📊 Ensemble Model Evaluation Metrics:")
        print(f"✅ MSE: {mse:.4f}")
        print(f"✅ RMSE: {rmse:.4f}")
        print(f"✅ MAE: {mae:.4f}")
        print(f"✅ R² Score: {r2:.4f}")

        plt.figure(figsize=(14, 7))
        plt.plot(y_test_inv, label="Actual Price", color="blue")
        plt.plot(y_pred_inv, label="Ensemble Predicted Price", color="orange")
        plt.title("Actual vs Ensemble Predicted Price")
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

    if X_train is not None and X_train.size > 0:
        predictor.train_and_evaluate(X_train, y_train, X_test, y_test)
    else:
        print("❌ No valid data. Exiting.")
