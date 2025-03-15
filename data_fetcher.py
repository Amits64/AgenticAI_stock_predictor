from binance.client import Client
import pandas as pd

# You will need your Binance API Key and Secret for this
api_key = 'FpmgOpAE2bez7ct136mQVPdRt6lbanMnuDK54iqP0l928bQ13pAN5VPKuqH71XK4'
api_secret = 'KWmXhMKxAvkofRbrsOrLusKLB351t6kBBAKzHDlOFd53y2uNnX88vtj73czZls3j'
client = Client(api_key, api_secret)


def fetch_historical_data(symbol='BTCUSDT', days=1825, interval='1d'):
    """
    Fetch historical crypto data from Binance.
    """
    # Fetch historical kline data from Binance API
    klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")

    # Convert data to pandas DataFrame
    df = pd.DataFrame(klines, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time',
                                       'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume',
                                       'Taker_buy_quote_asset_volume', 'Ignore'])

    # Convert the timestamp to datetime and set 'Date' as the index
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)  # Set the 'Date' column as the index

    # Include 'Volume' in the final DataFrame
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Include 'Volume' column

    # Convert numerical columns to float
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    # Verify 'Volume' is present and valid
    if 'Volume' not in df.columns or df['Volume'].isnull().any():
        raise ValueError("The 'Volume' column is missing or contains null values in the fetched data.")

    return df

def fetch_real_time_data(symbol='BTCUSDT'):
    """
    Fetch real-time crypto data from Binance.
    """
    ticker = client.get_symbol_ticker(symbol=symbol)
    ticker_24hr = client.get_ticker(symbol=symbol)

    return {
        "price": float(ticker['price']),
        "market_cap": "Not available directly from Binance",  # Binance does not provide market cap directly
        "high_24h": float(ticker_24hr['highPrice']),
        "low_24h": float(ticker_24hr['lowPrice']),
        "last_updated": pd.to_datetime(ticker_24hr['closeTime'], unit='ms')
    }


# Example usage
if __name__ == "__main__":
    try:
        # Fetch historical data for the last 365 days
        df = fetch_historical_data(symbol='BTCUSDT', days=1825, interval='1d')
        print("Historical Data:")
        print(df.head())

        # Fetch real-time data
        real_time_data = fetch_real_time_data(symbol='BTCUSDT')
        print("\nReal-Time Data:")
        print(real_time_data)
    except Exception as e:
        print(f"Error: {str(e)}")
