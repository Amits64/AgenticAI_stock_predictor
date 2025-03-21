import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

class SupertrendStrategyBacktester:
    def __init__(self, df, initial_capital=10000, slippage=0.0001, period=10, multiplier=3.0, changeATR=True):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.slippage = slippage
        self.period = period
        self.multiplier = multiplier
        self.changeATR = changeATR

    def supertrend(self, df):
        """
        Calculate the Supertrend indicator.
        ATR is approximated by True Range or by stddev if changeATR is True.
        """
        df = df.copy()
        # Calculate True Range and ATR
        df['tr'] = df['High'] - df['Low']
        df['atr'] = df['tr'].rolling(window=self.period).mean()
        if self.changeATR:
            df['atr'] = df['Close'].rolling(window=self.period).std()
        df['supertrend'] = np.nan  # Initialize supertrend column

        # Loop from index 'period' to the end
        for i in range(self.period, len(df)):
            if pd.isna(df['atr'].iloc[i]):
                continue
            atr = float(df['atr'].iloc[i])
            if pd.notna(df['supertrend'].iloc[i - 1]):
                prev_supertrend = float(df['supertrend'].iloc[i - 1])
            else:
                prev_supertrend = float(df['Close'].iloc[i]) - self.multiplier * atr
            current_close = float(df['Close'].iloc[i])
            up = current_close - self.multiplier * atr
            dn = current_close + self.multiplier * atr
            if current_close > prev_supertrend:
                df.loc[df.index[i], 'supertrend'] = max(up, prev_supertrend)
            else:
                df.loc[df.index[i], 'supertrend'] = min(dn, prev_supertrend)
        return df

    def generate_signals(self, df):
        """
        Generate trading signals using Supertrend:
        - Buy (signal = 1) if Close > Supertrend.
        - Sell (signal = -1) if Close < Supertrend.
        """
        df = df.copy()
        close_series = df['Close'].squeeze()
        df['supertrend'] = df['supertrend'].fillna(close_series)
        df['signal'] = 0
        df.loc[close_series > df['supertrend'], 'signal'] = 1
        df.loc[close_series < df['supertrend'], 'signal'] = -1
        df['position'] = df['signal'].diff()
        return df

    def calculate_sl_tp(self, df):
        """
        Calculate Stop Loss (SL) and Take Profit (TP):
        - SL is set as the previous candle's low.
        - TP is set as: TP = Close + 2 * (Close - previous candle's low)
          (i.e. a 1:2 risk–reward ratio).
        """
        df = df.copy()
        df['stop_loss'] = df['Low'].shift(1)
        close_series = df['Close'].squeeze()
        df['take_profit'] = close_series + 2 * (close_series - df['Low'].shift(1))
        return df

    def _apply_strategy(self):
        df = self.df.copy()
        df = self.supertrend(df)
        df = self.generate_signals(df)
        df = self.calculate_sl_tp(df)
        self.df = df
        return df

    def backtest(self):
        """
        Run the Supertrend strategy by applying the indicator,
        generating signals, and then computing performance metrics.
        """
        df = self.df.copy()
        # Reset index to ensure 'Date' is a column for plotting
        if "Date" not in df.columns:
            df = df.reset_index()
        df = self._apply_strategy()

        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            print("Error converting 'Date':", e)
            raise

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
        - Price along with Supertrend, Stop Loss, and Take Profit.
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
        # Plot price, supertrend, stop loss and take profit.
        plt.subplot(3, 1, 1)
        plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
        plt.plot(df['Date'], df['supertrend'], label='Supertrend', color='orange')
        plt.plot(df['Date'], df['stop_loss'], label='Stop Loss', color='red', linestyle='--')
        plt.plot(df['Date'], df['take_profit'], label='Take Profit', color='green', linestyle='--')
        plt.title('Price, Supertrend, SL & TP')
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
    """
    Helper function to run the Supertrend strategy backtest.
    """
    backtester = SupertrendStrategyBacktester(df)
    results, sharpe_ratio, max_drawdown = backtester.backtest()
    plot_url = backtester.plot_results()
    return results, sharpe_ratio, max_drawdown, plot_url
