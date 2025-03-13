# technical_analysis.py
import pandas as pd
import pandas_ta as ta

def add_technical_indicators(df: pd.DataFrame):
    """
    Adds various technical indicators to the DataFrame.
    """
    # Ensure relevant columns are numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')

    # Drop rows with NaN values after type conversion
    df = df.dropna(subset=['Close', 'High', 'Low'])

    # Simple and Exponential Moving Averages
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    # MACD
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
    return df

# Example usage
if __name__ == "__main__":
    # Get the current date and create a date range for the last year (365 days)
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=1)

    # Create the DataFrame with the date range from the last year
    df = pd.DataFrame({
        'Date': pd.date_range(start=start_date, end=end_date, freq='D'),
        'Open': pd.Series(range(365)),
        'High': pd.Series(range(365)),
        'Low': pd.Series(range(365)),
        'Close': pd.Series(range(365))
    })

    df = add_technical_indicators(df)
    print(df.head())