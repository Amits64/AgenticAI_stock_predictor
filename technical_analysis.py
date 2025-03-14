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

    # Check if 'Volume' exists before calculating VWAP
    if 'Volume' in df.columns:
        # Volume Weighted Average Price (VWAP)
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])

    # Average True Range (ATR)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # Volatility (Standard Deviation of Close prices)
    df['Volatility'] = df['Close'].rolling(window=20).std()

    return df


def normalize_data(df: pd.DataFrame):
    """
    Normalize the DataFrame to a range between 0 and 1.
    This step is crucial for the LSTM model to perform well.
    """
    # Use MinMaxScaler to normalize all columns except the 'Date' column
    scaler = MinMaxScaler(feature_range=(0, 1))
    columns_to_normalize = ['Close', 'Open', 'High', 'Low', 'SMA_20', 'EMA_20', 'RSI',
                            'MACD_12', 'MACD_26', 'MACD_9', 'BBL_20', 'BBM_20', 'BBU_20',
                            'ICH_SenkouSpanA', 'ICH_SenkouSpanB', 'Supertrend', 'VWAP', 'ATR', 'Volatility']

    # Check if 'Volume' exists before normalizing VWAP
    if 'Volume' in df.columns:
        columns_to_normalize.append('Volume')

    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    return df, scaler


# Example usage
if __name__ == "__main__":
    # Get the current date and create a date range for the last year (365 days)
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=1)

    # Create the DataFrame with the date range from the last year
    df = pd.DataFrame({
        'Date': pd.date_range(start=start_date, end=end_date, freq='D'),
        'Open': np.random.uniform(1000, 2000, 3650),
        'High': np.random.uniform(2000, 2500, 3650),
        'Low': np.random.uniform(800, 1000, 3650),
        'Close': np.random.uniform(1500, 2000, 3650),
        'Volume': np.random.randint(100000, 1000000, 3650)  # Volume data for VWAP
    })

    # Add technical indicators
    df = add_technical_indicators(df)

    # Normalize the data
    df, scaler = normalize_data(df)

    print(df.head())
