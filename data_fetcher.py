# data_fetcher.py
import requests
import pandas as pd

def fetch_historical_data(symbol='bitcoin', days='730', interval='daily'):
    """
    Fetch historical data for a given cryptocurrency from CoinGecko.
    """
    url = f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': interval
    }
    response = requests.get(url, params=params)
    data = response.json()

    if 'prices' not in data:
        raise ValueError(f"No data fetched for symbol: {symbol}")

    # Convert the data to a DataFrame
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['Open'] = df['Close']  # CoinGecko does not provide open prices
    df['High'] = df['Close']  # CoinGecko does not provide high prices
    df['Low'] = df['Close']   # CoinGecko does not provide low prices
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]

    return df

# Example usage
if __name__ == "__main__":
    df = fetch_historical_data(symbol='bitcoin', days='365', interval='daily')
    print(df.head())