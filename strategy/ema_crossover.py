# ema_crossover.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64


class EMACrossoverBacktester:
    def __init__(self, df, initial_capital=10000, slippage=0.0001, short_window=9, long_window=21):
        """
        Initialize the backtester with the necessary parameters.
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.slippage = slippage
        self.short_window = short_window
        self.long_window = long_window

    def _apply_strategy(self):
        """
        Compute the short and long EMAs, generate signals, and calculate SL/TP.
        Signal: Buy (1) when EMA(short) > EMA(long), otherwise 0.

        Stop Loss is set as the previous candle's low.
        Take Profit is calculated as: Close + 2 * (Close - previous candle's low).
        """
        df = self.df.copy()
        df['EMA_short'] = df['Close'].ewm(span=self.short_window, adjust=False).mean()
        df['EMA_long'] = df['Close'].ewm(span=self.long_window, adjust=False).mean()

        # Generate trading signals based on the crossover
        df['signal'] = np.where(df['EMA_short'] > df['EMA_long'], 1, 0)
        df['position'] = df['signal'].diff()  # Change in signal indicates a trade

        # Calculate Stop Loss (previous candle's low) and Take Profit (1:2 risk-reward)
        df['stop_loss'] = df['Low'].shift(1)
        df['take_profit'] = df['Close'] + 2 * (df['Close'] - df['stop_loss'])

        self.df = df
        return df

    def backtest(self):
        """
        Run the EMA crossover strategy:
        - Apply strategy (EMAs, signals, SL/TP).
        - Compute daily returns, cumulative return, portfolio value, Sharpe ratio, and drawdown.
        """
        df = self._apply_strategy()
        # Ensure 'Date' is a column for plotting
        if "Date" not in df.columns:
            df = df.reset_index()
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            print("Error converting 'Date':", e)
            raise

        # Compute returns
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
        Generate a plot that includes:
        - Price with the short and long EMAs, SL, and TP.
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

        plt.figure(figsize=(14, 10))

        # Subplot 1: Price, EMAs, Stop Loss, and Take Profit
        plt.subplot(3, 1, 1)
        plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
        plt.plot(df['Date'], df['EMA_short'], label=f'EMA {self.short_window}', color='green')
        plt.plot(df['Date'], df['EMA_long'], label=f'EMA {self.long_window}', color='red')
        plt.plot(df['Date'], df['stop_loss'], label='Stop Loss', color='orange', linestyle='--')
        plt.plot(df['Date'], df['take_profit'], label='Take Profit', color='purple', linestyle='--')
        plt.title('Price & EMA Crossover with SL/TP')
        plt.legend()

        # Subplot 2: Cumulative return
        plt.subplot(3, 1, 2)
        plt.plot(df['Date'], df['cumulative_return'], label='Cumulative Return', color='purple')
        plt.title('Cumulative Strategy Return')
        plt.legend()

        # Subplot 3: Drawdown
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
    """
    Helper function to run the EMA Crossover strategy backtest.
    """
    backtester = EMACrossoverBacktester(df)
    results, sharpe_ratio, max_drawdown = backtester.backtest()
    plot_url = backtester.plot_results()
    return results, sharpe_ratio, max_drawdown, plot_url
