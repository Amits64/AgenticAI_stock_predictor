import os
import numpy as np
import pandas as pd
import pickle
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import joblib
import pandas_ta as ta


class ModelManager:
    """
    Manages AI models: loads a pre-trained LSTM model and an ensemble stacking model,
    refits the scaler, processes input data, and outputs predictions along with a plot.
    """

    def __init__(self,
                 ensemble_model_filename='ensemble_model.pkl',
                 lstm_model_filename='model_b_20250322-181312.keras',
                 scaler_filename='scaler.pkl',
                 data_filename='raw_data.csv',
                 feature_names_filename='features.pkl'):
        self.ensemble_model_filename = ensemble_model_filename
        self.lstm_model_filename = lstm_model_filename
        self.scaler_filename = scaler_filename
        self.data_filename = data_filename
        self.feature_names_filename = feature_names_filename

        # Define the feature set used during training (11 features)
        self.features = [
            'Close', 'SMA_20', 'EMA_20', 'RSI_14',
            'MACD', 'BBL', 'BBM', 'BBU', 'ADX', 'CCI', 'OBV'
        ]

        # Load ensemble model and associated feature names
        self.ensemble_model = self.load_ensemble_model()
        # Load LSTM model and scaler
        self.lstm_model = self.load_lstm_model()
        self.scaler = self.load_scaler()

    def load_ensemble_model(self):
        """Load the pre-trained ensemble model and verify feature set."""
        if os.path.exists(self.ensemble_model_filename) and os.path.exists(self.feature_names_filename):
            print(f"✅ Loading ensemble model from '{self.ensemble_model_filename}'...")
            model = joblib.load(self.ensemble_model_filename)
            with open(self.feature_names_filename, 'rb') as f:
                loaded_features = pickle.load(f)
            if len(loaded_features) != len(self.features):
                print("🔄 Feature mismatch detected. Using updated feature set.")
                self.ensemble_features = self.features
            else:
                self.ensemble_features = loaded_features
            return model
        else:
            print("❌ Ensemble model or feature file not found.")
            return None

    def load_lstm_model(self):
        """Load the pre-trained LSTM model."""
        if os.path.exists(self.lstm_model_filename):
            try:
                lstm_model = load_model(self.lstm_model_filename)
                print("✅ LSTM model loaded successfully.")
                return lstm_model
            except Exception as e:
                print(f"❌ Error loading LSTM model: {e}")
                return None
        else:
            print("❌ LSTM model file not found.")
            return None

    def load_scaler(self):
        """Load the pre-fitted scaler."""
        if os.path.exists(self.scaler_filename):
            try:
                with open(self.scaler_filename, "rb") as f:
                    scaler = pickle.load(f)
                print("✅ Scaler loaded successfully.")
                return scaler
            except Exception as e:
                print(f"❌ Error loading scaler: {e}")
                return None
        else:
            print("❌ Scaler file not found.")
            return None

    def load_data_from_csv(self, file_path=None):
        """Load CSV data, sort by Date, and add technical indicators."""
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
            print(f"📊 Data with indicators loaded: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading CSV data: {e}")
            return None

    def add_technical_indicators(self, df):
        """Add technical indicators to DataFrame using the predefined feature set."""
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

    def create_lstm_sequences(self, data, seq_length=60):
        """Create LSTM sequences from data with given sequence length."""
        sequences = []
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            sequences.append(seq)
        return np.array(sequences)

    def plot_predictions(self, y_actual, lstm_pred, ensemble_pred):
        """Plot actual vs. predictions and return a base64 image URL."""
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(y_actual, label="Actual Price", color="blue")
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
        """Prepare data, get predictions from the LSTM and ensemble models, and return results."""
        df = self.load_data_from_csv()
        if df is None or df.empty:
            return {"error": "No valid data for prediction."}

        # Use the ensemble feature set to create input data
        data_np = df[self.features].to_numpy().astype(np.float32)
        data_scaled = self.scaler.transform(data_np)

        # Create LSTM sequences (assumes a sequence length of 60)
        lstm_X = self.create_lstm_sequences(data_scaled, seq_length=60)
        lstm_preds = self.lstm_model.predict(lstm_X).flatten()

        # Ensemble predictions: use the last N rows of scaled data, where N equals the number of LSTM predictions
        ensemble_preds = self.ensemble_model.predict(data_scaled[-len(lstm_preds):])

        mse_val = mean_squared_error(lstm_preds, ensemble_preds)
        y_actual = df['Close'].values[-len(lstm_preds):]
        predict_plot = self.plot_predictions(y_actual, lstm_preds, ensemble_preds)

        return {
            "ensemble_prediction": float(ensemble_preds[-1]),
            "lstm_prediction": float(lstm_preds[-1]),
            "mse": mse_val,
            "predict_url": predict_plot
        }


if __name__ == "__main__":
    manager = ModelManager()
    results = manager.predict()
    print("\n📊 Prediction Results:")
    print(results)
