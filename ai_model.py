import os
import numpy as np
import pandas as pd
import pickle
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

class ModelManager:
    """
    Manages AI models: LSTM and ensemble, handles predictions, and plots the graph for 'predict_url'.
    """

    def __init__(self, model_filename='ensemble_model.pkl',
                 lstm_filename='model_20250322-041338.keras',
                 scaler_filename='scaler.pkl',
                 data_filename='raw_data.csv',
                 feature_names_file='features.pkl'):
        self.model_filename = model_filename
        self.lstm_filename = lstm_filename
        self.scaler_filename = scaler_filename
        self.data_filename = data_filename
        self.feature_names_file = feature_names_file

        # Define the updated feature list (11 features)
        self.new_features = [
            'Close', 'SMA_20', 'EMA_20', 'RSI_14',
            'MACD', 'BBL', 'BBM', 'BBU', 'ADX', 'CCI', 'OBV'
        ]

        # Load or train the ensemble model with updated feature names
        self.ensemble_model = self.load_or_train_ensemble_model()

        # Load the LSTM model and scaler
        self.lstm_model, self.scaler = self.load_lstm_model()

    def load_or_train_ensemble_model(self):
        """Load or train the ensemble model."""
        # Check if the model and feature files exist
        if os.path.exists(self.model_filename) and os.path.exists(self.feature_names_file):
            print(f"✅ Loading pre-trained ensemble model from '{self.model_filename}'...")
            model = joblib.load(self.model_filename)
            with open(self.feature_names_file, 'rb') as f:
                loaded_features = pickle.load(f)
            # If the loaded feature list doesn't match our new feature list, retrain the model.
            if len(loaded_features) != len(self.new_features):
                print("🔄 Feature shape mismatch detected (loaded features vs. new features).")
                print("🚀 Retraining ensemble model with updated feature set...")
                os.remove(self.model_filename)
                os.remove(self.feature_names_file)
                return self.train_new_ensemble_model()
            else:
                self.ensemble_features = loaded_features
                return model
        else:
            print("🚀 Training new ensemble model...")
            return self.train_new_ensemble_model()

    def train_new_ensemble_model(self):
        """Train a new ensemble model with updated features."""
        self.ensemble_features = self.new_features
        # Dummy data (replace with real windowed data when available)
        X_train = np.random.random((100, len(self.ensemble_features)))
        y_train = np.random.random(100)
        model = self.train_ensemble_model(X_train, y_train)
        # Save model and features
        joblib.dump(model, self.model_filename)
        with open(self.feature_names_file, 'wb') as f:
            pickle.dump(self.ensemble_features, f)
        return model

    def load_lstm_model(self):
        """Load the LSTM model and refit scaler."""
        try:
            lstm_model = load_model(self.lstm_filename)
            self.refit_scaler()
            print("✅ LSTM model and scaler loaded successfully.")
            return lstm_model, self.scaler
        except Exception as e:
            print(f"❌ Error loading LSTM model: {e}")
            return None, None

    def refit_scaler(self):
        """Refit the MinMaxScaler with current feature names."""
        df = self.load_data_from_csv()
        if df is None:
            return
        df = self.add_technical_indicators(df)
        # Update ensemble_features if needed
        if not all(feature in df.columns for feature in self.ensemble_features):
            print("🔄 Updating ensemble features to match current data columns.")
            self.ensemble_features = self.new_features
        self.scaler = MinMaxScaler()
        self.scaler.fit(df[self.ensemble_features])
        with open(self.scaler_filename, "wb") as f:
            pickle.dump(self.scaler, f)

    def train_ensemble_model(self, X, y):
        """Train an ensemble stacking model."""
        estimators = [
            ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1)),
            ('ridge', Ridge(alpha=1.0)),
            ('knn', KNeighborsRegressor(n_neighbors=5)),
            ('svr', SVR(kernel='rbf'))
        ]
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=XGBRegressor(n_estimators=100, learning_rate=0.1)
        )
        model.fit(X, y)
        print("✅ Ensemble model trained successfully.")
        return model

    def load_data_from_csv(self, file_path=None):
        """Load and validate CSV data."""
        if file_path is None:
            file_path = self.data_filename
        if not os.path.exists(file_path):
            print(f"❌ CSV file '{file_path}' does not exist.")
            return None
        try:
            print(f"✅ Loading data from '{file_path}'...")
            df = pd.read_csv(file_path)
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                print("❌ Missing required columns in CSV.")
                return None
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values('Date', inplace=True)
            df.dropna(inplace=True)
            df = self.add_technical_indicators(df)
            print(f"📊 Data with indicators loaded successfully: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading CSV data: {e}")
            return None

    def add_technical_indicators(self, df):
        """Add technical indicators to the DataFrame using 11 features."""
        import pandas_ta as ta
        print("🔧 Adding technical indicators...")
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        bb = ta.bbands(df['Close'], length=20, std=2)
        df['BBL'] = bb['BBL_20_2.0']
        df['BBM'] = bb['BBM_20_2.0']
        df['BBU'] = bb['BBU_20_2.0']
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = adx['ADX_14']
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df.dropna(inplace=True)
        return df

    def create_sequences(self, data, seq_length=50):
        """Create valid LSTM sequences."""
        sequences = []
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            sequences.append(seq)
        return np.array(sequences)

    def plot_predictions(self, y_test, lstm_pred, ensemble_pred):
        fig, ax = plt.subplots(figsize=(14, 7))
        plt.style.use("seaborn-v0_8")
        ax.plot(y_test, label="Actual Price", color="blue")
        ax.plot(lstm_pred, label="LSTM Prediction", color="orange")
        ax.plot(ensemble_pred, label="Ensemble Prediction", color="green")
        ax.set_title("Actual vs LSTM vs Ensemble Predictions")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{plot_url}"

    def predict(self):
        df = self.load_data_from_csv()
        if df is None or df.empty:
            return {"error": "No valid data for prediction."}
        # Prepare data using the 11 ensemble features
        data_np = df[self.ensemble_features].to_numpy().astype(np.float32)
        data_scaled = self.scaler.transform(data_np)
        # Ensure the sequence length matches the model's training sequence length, e.g., 60
        lstm_X = self.create_sequences(data_scaled, seq_length=60)
        lstm_preds = self.lstm_model.predict(lstm_X).flatten()
        # Ensure ensemble predictions use the last N rows where N equals the number of LSTM predictions
        ensemble_preds = self.ensemble_model.predict(data_scaled[-len(lstm_preds):])
        mse = mean_squared_error(lstm_preds, ensemble_preds)
        y_test = df['Close'].values[-len(lstm_preds):]
        predict_plot = self.plot_predictions(y_test, lstm_preds, ensemble_preds)
        return {
            "ensemble_prediction": float(ensemble_preds[-1]),
            "lstm_prediction": float(lstm_preds[-1]),
            "mse": mse,
            "predict_url": predict_plot
        }

if __name__ == "__main__":
    manager = ModelManager()
    results = manager.predict()
    print("\n📊 Prediction Results:")
    print(results)
