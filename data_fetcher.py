from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from config import Config
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

class BinanceHistoricalData:
    """ Fetch historical data from Binance """
    def __init__(self, api_key, api_secret, symbol='BTCUSDT', days=1825, interval='1d'):
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        self.days = days
        self.interval = interval
        self.df = pd.DataFrame()

    def fetch_historical_data(self):
        """ Fetch crypto data from Binance and convert to DataFrame """
        try:
            klines = self.client.get_historical_klines(self.symbol, self.interval, f"{self.days} day ago UTC")

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time',
                'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume',
                'Taker_buy_quote_asset_volume', 'Ignore'
            ])

            # Format DataFrame
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

            # Convert to float
            df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

            # Add synthetic volume if missing
            if df['Volume'].isnull().all():
                print("⚠️ Missing volume data. Adding synthetic volume.")
                df['Volume'] = pd.Series(pd.np.random.randint(1000, 10000, size=len(df)))

            self.df = df
            df.to_csv("raw_data.csv", index=False)
            print(f"✅ Data saved to raw_data.csv with {len(df)} rows")
            return df

        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return pd.DataFrame()

    def generate_plot_url(self):
        """ Generate base64 URL for the plot """
        if self.df.empty:
            print("⚠️ No data available for plotting.")
            return ""

        plt.ioff()  # Disable interactive mode
        plt.figure(figsize=(14, 6))
        plt.style.use("seaborn-v0_8")
        plt.plot(self.df['Date'], self.df['Close'], label='Close Price', color='blue')
        plt.title(f'Historical Close Price for {self.symbol}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.grid(True)
        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{plot_url}"


# ---------------------------
# 🚀 Main Execution
# ---------------------------
if __name__ == "__main__":
    api_key = Config.BINANCE_API_KEY
    api_secret = Config.BINANCE_API_SECRET

    fetcher = BinanceHistoricalData(api_key, api_secret, symbol='BTCUSDT', days=1825, interval='1d')
    fetcher.fetch_historical_data()

    fetch_url = fetcher.generate_plot_url()
    print(f"✅ Plot URL: {fetch_url}")
