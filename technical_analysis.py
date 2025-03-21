import pandas as pd
import numpy as np
import pickle
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler


class TechnicalAnalysis:
    """ Class for applying indicators and creating windowed data """

    def __init__(self, csv_file="raw_data.csv", window_size=50):
        """ Initialize with CSV file and window size """
        self.df = pd.read_csv(csv_file, parse_dates=['Date'])
        self.window_size = window_size

    # ---------------------------
    # 📊 Add Technical Indicators
    # ---------------------------
    def add_indicators(self):
        """ Add technical indicators """
        print("\n✅ Adding Technical Indicators...")

        # Ensure volume exists or add synthetic volume
        if 'Volume' not in self.df.columns:
            print("⚠️ Adding synthetic 'Volume' column.")
            self.df['Volume'] = np.random.randint(1000, 10000, size=len(self.df))

        # Apply indicators
        self.df.ta.ema(length=50, append=True)
        self.df.ta.ema(length=200, append=True)
        self.df.ta.rsi(length=14, append=True)
        self.df.ta.macd(fast=12, slow=26, signal=9, append=True)
        self.df.ta.bbands(length=20, std=2, append=True)
        self.df.ta.adx(length=14, append=True)
        self.df.ta.atr(length=14, append=True)

        # Fill missing values
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(method='bfill', inplace=True)

        print("\n✅ Indicators Applied Successfully!")

    # ---------------------------
    # 🔥 Scale Data and Create Windows
    # ---------------------------
    def scale_and_window(self):
        """ Scale data and create windowed sequences """
        print("\n✅ Scaling and creating windowed sequences...")

        # ✅ Use 9 features for consistency with your AI model
        features = [
            'Close',           # 1
            'EMA_50',          # 2
            'EMA_200',         # 3
            'RSI_14',          # 4
            'MACD_12_26_9',    # 5
            'BBL_20_2.0',      # 6 Lower Bollinger Band
            'BBM_20_2.0',      # 7 Middle Bollinger Band
            'BBU_20_2.0',      # 8 Upper Bollinger Band
            'ADX_14'           # 9
        ]

        # ✅ Ensure all features are present
        missing_features = [f for f in features if f not in self.df.columns]
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            return

        # ✅ Drop NaNs before scaling
        self.df.dropna(inplace=True)

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.df[features])

        # ✅ Create sequences
        X, Y, dates = [], [], []

        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i - self.window_size:i])
            Y.append(scaled_data[i, 0])  # Target: Close price
            dates.append(self.df['Date'].iloc[i])

        X, Y = np.array(X), np.array(Y)

        # ✅ Save windowed data
        windowed_df = pd.DataFrame({
            'Date': dates,
            'X': list(X),
            'Y': Y
        })

        with open("windowed_data.pkl", "wb") as f:
            pickle.dump(windowed_df, f)

        print(f"✅ Windowed data saved as 'windowed_data.pkl' with {len(Y)} samples")

    # ---------------------------
    # 📥 Save Final Results
    # ---------------------------
    def save_results(self, output_file="technical_analysis_results.csv"):
        """ Save the final DataFrame to CSV """
        self.df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to {output_file}")


# ---------------------------
# 🚀 Main Execution
# ---------------------------
if __name__ == "__main__":
    ta = TechnicalAnalysis(csv_file="raw_data.csv", window_size=50)

    ta.add_indicators()
    ta.scale_and_window()
    ta.save_results()
