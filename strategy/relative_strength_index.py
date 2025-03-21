import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

class RSIStrategyBacktester:
    def __init__(self, df, initial_capital=10000, slippage=0.0001, rsi_period=14, overbought=60, oversold=25):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.slippage = slippage
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold

    def rsi(self, df, period):
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df = df.ffill()  # Forward fill missing values
        return df

    def calculate_sl_tp(self, df):
        # Use the previous low as stop loss and compute take profit as 1:2 risk/reward.
        df['stop_loss'] = df['Low'].shift(1)
        close_series = df['Close']
        df['take_profit'] = close_series + 2 * (close_series - df['stop_loss'])
        return df

    def _apply_strategy(self):
        df = self.df
        # Calculate RSI and SL/TP values
        df = self.rsi(df, self.rsi_period)
        df = self.calculate_sl_tp(df)
        # Generate trading signals: Buy if RSI is below the oversold threshold; Sell if above overbought.
        df['signal'] = 0
        df.loc[df['rsi'] < self.oversold, 'signal'] = 1
        df.loc[df['rsi'] > self.overbought, 'signal'] = -1
        df['position'] = df['signal'].diff()
        self.df = df

    def backtest(self):
        self._apply_strategy()
        df = self.df
        # Ensure that a "Date" column exists and is in datetime format.
        if "Date" not in df.columns:
            df = df.reset_index()
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            print("Error converting 'Date':", e)
            raise

        # Calculate returns and performance metrics.
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
        df = self.df
        if "Date" not in df.columns:
            df = df.reset_index()
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            print("Error converting 'Date' in plot_results:", e)
            raise

        plt.figure(figsize=(14, 10))
        # Plot price and RSI indicator.
        plt.subplot(3, 1, 1)
        plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
        plt.plot(df['Date'], df['rsi'], label='RSI', color='orange')
        plt.axhline(self.overbought, color='red', linestyle='--', label='Overbought')
        plt.axhline(self.oversold, color='green', linestyle='--', label='Oversold')
        plt.title('Price & RSI')
        plt.legend()

        # Plot cumulative return.
        plt.subplot(3, 1, 2)
        plt.plot(df['Date'], df['cumulative_return'], label='Cumulative Return', color='purple')
        plt.title('Cumulative Strategy Return')
        plt.legend()

        # Plot drawdown.
        plt.subplot(3, 1, 3)
        plt.fill_between(df['Date'], df['drawdown'], color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.xlabel('Date')

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        full_plot_url = f"data:image/png;base64,{plot_b64}"
        return full_plot_url

def backtest_strategy(df):
    backtester = RSIStrategyBacktester(df)
    results, sharpe_ratio, max_drawdown = backtester.backtest()
    plot_url = backtester.plot_results()
    return results, sharpe_ratio, max_drawdown, plot_url
