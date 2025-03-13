# data_fetcher.py
import requests
import pandas as pd


def fetch_historical_data(symbol='bitcoin', days='365', interval='daily'):
   """
   Fetch historical crypto data from CoinGecko.
   """
   url = f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart'
   params = {'vs_currency': 'usd', 'days': days, 'interval': interval}
   response = requests.get(url, params=params)
   data = response.json()
   if 'prices' not in data:
       raise ValueError(f"No data fetched for {symbol}")
   df = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
   df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
   df['Open'] = df['Close']
   df['High'] = df['Close']
   df['Low'] = df['Close']
   df = df[['Date', 'Open', 'High', 'Low', 'Close']]
   return df


def fetch_real_time_data(symbol='bitcoin'):
   """
   Fetch real-time crypto data from CoinGecko.
   """
   url = f'https://api.coingecko.com/api/v3/simple/price'
   params = {'ids': symbol, 'vs_currencies': 'usd', 'include_market_cap': 'true', 'include_24hr_high': 'true', 'include_24hr_low': 'true', 'include_last_updated_at': 'true'}
   response = requests.get(url, params=params)
   data = response.json()
   if symbol not in data:
       raise ValueError(f"No real-time data available for {symbol}")
   return {
       "price": data[symbol]['usd'],
       "market_cap": data[symbol]['usd_market_cap'],
       "high_24h": data[symbol]['usd_24h_high'],
       "low_24h": data[symbol]['usd_24h_low'],
       "last_updated": pd.to_datetime(data[symbol]['last_updated_at'], unit='s')
   }


# Example usage
if __name__ == "__main__":
    df = fetch_historical_data(symbol='bitcoin', days='365', interval='daily')
    print(df.head())