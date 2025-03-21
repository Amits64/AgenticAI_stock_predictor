import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

class MultiFactorStrategyBacktester:
    def __init__(self, df, initial_capital=10000, slippage=0.0001):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.slippage = slippage

    def calculate_indicators(self, df):
        """
        Calculate a set of technical indicators:
        - Momentum: 20-day SMA.
        - Trend Filter: 50-day SMA.
        - Volatility: 14-day ATR approximation.
        - Mean reversion: Bollinger Bands (20-day SMA ± 2 std dev).
        """
        df = df.copy()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['ATR'] = df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()
        df['std20'] = df['Close'].rolling(window=20).std()
        df['UpperBand'] = df['SMA20'] + 2 * df['std20']
        df['LowerBand'] = df['SMA20'] - 2 * df['std20']
        return df

    def generate_signals(self, df):
        """
        Generate signals based on:
        - Long entry if: Price is above SMA50, above SMA20, and below UpperBand.
        - Exit signal if: Price is below SMA20 or above UpperBand.
        """
        df = df.copy()
        df['signal'] = 0
        close = df['Close'].squeeze()
        sma20 = df['SMA20'].squeeze()
        sma50 = df['SMA50'].squeeze()
        upperBand = df['UpperBand'].squeeze()
        # Entry condition: long if above both SMAs and below the UpperBand.
        condition_entry = (close > sma50) & (close > sma20) & (close < upperBand)
        # Exit condition: sell if price is below SMA20 or above UpperBand.
        condition_exit = (close < sma20) | (close > upperBand)
        df.loc[condition_entry, 'signal'] = 1
        df.loc[condition_exit, 'signal'] = -1
        df['position'] = df['signal'].diff()
        return df

    def dynamic_stop_take(self, df):
        """
        Set dynamic stop-loss and take-profit levels using ATR:
        - Stop Loss: Close - (1.5 * ATR).
        - Take Profit: Close + (3 * ATR).
        """
        df = df.copy()
        atr = df['ATR'].squeeze()
        close = df['Close'].squeeze()
        df['stop_loss'] = close - 1.5 * atr
        df['take_profit'] = close + 3 * atr
        return df

    def backtest(self):
        """
        Run the multi-factor strategy:
        - Calculate indicators.
        - Generate signals.
        - Compute stop loss/take profit.
        - Calculate returns, cumulative return, and performance metrics.
        """
        df = self.df.copy()
        if "Date" not in df.columns:
            df = df.reset_index()
        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
        df = self.dynamic_stop_take(df)
        df['daily_return'] = df['Close'].pct_change()
        df['strategy_return'] = df['daily_return'] * df['signal'].shift(1)
        df['cumulative_return'] = (1 + df['strategy_return']).cumprod() - 1
        df['portfolio_value'] = self.initial_capital * (1 + df['strategy_return']).cumprod()
        df['slippage_adjusted'] = df['Close'] * (1 + self.slippage)
        risk_free_rate = 0.01
        df['excess_return'] = df['strategy_return'] - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * df['excess_return'].mean() / df['excess_return'].std()
        df['cumulative_max'] = df['cumulative_return'].cummax()
        df['drawdown'] = df['cumulative_return'] - df['cumulative_max']
        max_drawdown = df['drawdown'].min()
        self.df = df
        return df, sharpe_ratio, max_drawdown

    def plot_results(self):
        """
        Generate plots for:
        - Price and indicators (SMA20, SMA50, Bollinger Bands).
        - Cumulative return.
        - Drawdown.
        Returns a base64-encoded PNG image URL.
        """
        df = self.df.copy()
        if "Date" not in df.columns:
            df = df.reset_index()
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            print("Error converting 'Date' in plot_results:", e)
            raise

        plt.style.use("seaborn-v0_8")
        plt.figure(figsize=(14, 12))
        # Price and indicators plot.
        plt.subplot(3, 1, 1)
        plt.plot(df['Date'], df['Close'], label='Close', color='blue')
        plt.plot(df['Date'], df['SMA20'], label='SMA20', color='orange', linestyle='--')
        plt.plot(df['Date'], df['SMA50'], label='SMA50', color='green', linestyle='--')
        plt.fill_between(df['Date'], df['UpperBand'], df['LowerBand'], color='gray', alpha=0.3, label='Bollinger Bands')
        plt.title('Multi-Factor Strategy: Price & Indicators')
        plt.legend()

        # Cumulative return plot.
        plt.subplot(3, 1, 2)
        plt.plot(df['Date'], df['cumulative_return'], label='Cumulative Return', color='purple')
        plt.title('Cumulative Strategy Return')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()

        # Drawdown plot.
        plt.subplot(3, 1, 3)
        plt.fill_between(df['Date'], df['drawdown'], color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        full_plot_url = f"data:image/png;base64,{plot_b64}"
        return full_plot_url

def backtest_strategy(df):
    """
    Helper function to run the Multi-Factor strategy backtest.
    """
    backtester = MultiFactorStrategyBacktester(df)
    results, sharpe_ratio, max_drawdown = backtester.backtest()
    plot_url = backtester.plot_results()
    return results, sharpe_ratio, max_drawdown, plot_url
