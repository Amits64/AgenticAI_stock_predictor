import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def add_technical_indicators(df: pd.DataFrame):
    """
    Adds various technical indicators to the DataFrame, including Ichimoku, Supertrend, and more.
    """
    # Ensure relevant columns are numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # Optionally, fill NaN values instead of dropping them (e.g., forward fill or use another method)
    df = df.fillna(method='ffill')  # Or use 'bfill' for backward fill

    # Simple and Exponential Moving Averages
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['EMA_20'] = ta.ema(df['Close'], length=20)

    # RSI (Relative Strength Index)
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # MACD (Moving Average Convergence Divergence)
    macd = ta.macd(df['Close'])
    if isinstance(macd, pd.DataFrame):
        df = pd.concat([df, macd], axis=1)

    # Bollinger Bands
    bbands = ta.bbands(df['Close'])
    if isinstance(bbands, pd.DataFrame):
        df = pd.concat([df, bbands], axis=1)

    # Ichimoku Cloud (default settings)
    ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])
    if isinstance(ichimoku, pd.DataFrame):
        df = pd.concat([df, ichimoku], axis=1)

    # Supertrend
    supertrend = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3.0)
    if isinstance(supertrend, pd.Series):
        df['Supertrend'] = supertrend

    # Ensure 'Date' exists and is correctly formatted
    if 'Date' not in df.columns:
        if 'timestamp' in df.columns:  # Assuming the data may have a 'timestamp' column
            df['Date'] = pd.to_datetime(df['timestamp'], unit='s')  # Convert from Unix timestamp to DateTime
        else:
            raise ValueError("The DataFrame must contain a 'Date' column or a 'timestamp' column.")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Ensure 'Date' is in datetime format

    # Check the 'Date' column before proceeding
    print(f"Date column after conversion:\n{df['Date'].head()}")

    # Set the 'Date' column as the index and ensure it's sorted by date
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)  # Ensure the data is sorted by Date

    # Volume Weighted Average Price (VWAP)
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])

    # Average True Range (ATR)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # Volatility (Standard Deviation of Close prices)
    df['Volatility'] = df['Close'].rolling(window=20).std()

    # Check the columns after technical indicators are added
    print("Columns in the DataFrame after adding technical indicators:")
    print(df.columns)

    return df


def normalize_data(df: pd.DataFrame):
    """
    Normalize the DataFrame to a range between 0 and 1.
    This step is crucial for the LSTM model to perform well.
    """
    # Use MinMaxScaler to normalize all columns except the 'Date' column
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Update the columns to match the correct names based on printed output
    columns_to_normalize = ['Close', 'Open', 'High', 'Low', 'SMA_20', 'EMA_20', 'RSI',
                            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'BBL_5_2.0',
                            'BBM_5_2.0', 'BBU_5_2.0', 'VWAP', 'ATR', 'Volatility']

    # Normalize the relevant columns
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    return df, scaler


# Example usage
if __name__ == "__main__":
    # Get the current date and create a date range for the last year (365 days)
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=1)

    # Create the DataFrame with the date range from the last year (ensure the length matches num_days)
    num_days = 365  # Length of the time period (365 days)
    df = pd.DataFrame({
        'Date': pd.date_range(start=start_date, periods=num_days, freq='D'),  # Ensure periods=num_days
        'Open': np.random.uniform(1000, 2000, num_days),
        'High': np.random.uniform(2000, 2500, num_days),
        'Low': np.random.uniform(800, 1000, num_days),
        'Close': np.random.uniform(1500, 2000, num_days),
        'Volume': np.random.randint(100000, 1000000, num_days)  # Volume data for VWAP
    })

    # Add technical indicators
    df = add_technical_indicators(df)

    # Normalize the data
    df, scaler = normalize_data(df)

    print(df.head())
