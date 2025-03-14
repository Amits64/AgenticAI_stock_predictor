import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Ichimoku Cloud Calculation
def ichimoku_cloud(df):
    # Tenkan-sen (Conversion Line)
    df['tenkan_sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    # Kijun-sen (Base Line)
    df['kijun_sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    # Senkou Span A (Leading Span A)
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    # Senkou Span B (Leading Span B)
    df['senkou_span_b'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    # Chikou Span (Lagging Line)
    df['chikou_span'] = df['Close'].shift(-26)

    return df

# Supertrend Calculation
def supertrend(df, period=10, multiplier=3.0):
    df['atr'] = df['High'].rolling(window=period).max() - df['Low'].rolling(window=period).min()
    df['supertrend'] = 0.0
    for i in range(period, len(df)):
        hl = df['High'][i] - df['Low'][i]
        atr = df['atr'][i]
        if df['Close'][i] > df['supertrend'][i - 1]:
            df.loc[i, 'supertrend'] = df['Close'][i] - multiplier * atr
        else:
            df.loc[i, 'supertrend'] = df['Close'][i] + multiplier * atr
    return df

# Trading Strategy: Ichimoku Cloud + Supertrend
def ichimoku_supertrend_strategy(df):
    df = ichimoku_cloud(df)
    df = supertrend(df)

    df['signal'] = 0  # Default to no position

    # Buy Signal: Price above the cloud and Tenkan-sen above Kijun-sen, Supertrend below price
    df.loc[(df['Close'] > df['senkou_span_a']) & (df['Close'] > df['senkou_span_b']) &
           (df['tenkan_sen'] > df['kijun_sen']) & (df['Close'] > df['supertrend']), 'signal'] = 1

    # Sell Signal: Price below the cloud and Tenkan-sen below Kijun-sen, Supertrend above price
    df.loc[(df['Close'] < df['senkou_span_a']) & (df['Close'] < df['senkou_span_b']) &
           (df['tenkan_sen'] < df['kijun_sen']) & (df['Close'] < df['supertrend']), 'signal'] = -1

    df['position'] = df['signal'].diff()

    return df

# Backtesting the Strategy
def backtest_strategy(df, initial_capital=10000, position_size=1, slippage=0.0001):
    df = ichimoku_supertrend_strategy(df)

    # Calculate daily returns
    df['daily_return'] = df['Close'].pct_change()

    # Strategy return is the daily return when in position (shifted by 1 to avoid lookahead bias)
    df['strategy_return'] = df['daily_return'] * df['signal'].shift(1)

    # Cumulative strategy return
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod() - 1

    # Add Portfolio value assuming an initial capital
    df['portfolio_value'] = initial_capital * (1 + df['strategy_return']).cumprod()

    # Position size calculation (number of shares)
    df['position_size'] = (df['portfolio_value'] * position_size) / df['Close']

    # Slippage effect: slightly adjust for transaction cost or price slippage
    df['slippage_adjusted'] = df['Close'] * (1 + slippage)

    # Additional Performance Metrics:
    # Sharpe ratio calculation
    risk_free_rate = 0.01
    df['excess_return'] = df['strategy_return'] - risk_free_rate / 252  # Annualize over 252 trading days
    sharpe_ratio = np.sqrt(252) * df['excess_return'].mean() / df['excess_return'].std()

    # Max Drawdown calculation
    df['cumulative_max'] = df['cumulative_return'].cummax()
    df['drawdown'] = df['cumulative_return'] - df['cumulative_max']
    max_drawdown = df['drawdown'].min()

    # Display Performance Metrics
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Max Drawdown: {max_drawdown:.4f}")

    return df[['Date', 'Close', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'supertrend', 'signal',
               'cumulative_return', 'portfolio_value', 'position_size', 'slippage_adjusted',
               'drawdown']], sharpe_ratio, max_drawdown

# Plotting Function (same as original)
def plot_backtest(df):
    plt.figure(figsize=(14, 10))

    # Plotting Closing Price, Ichimoku Cloud, and Supertrend
    plt.subplot(3, 1, 1)
    plt.plot(df['Date'], df['Close'], label='Closing Price', color='blue', alpha=0.7)
    plt.plot(df['Date'], df['tenkan_sen'], label='Tenkan-sen', color='green', alpha=0.7)
    plt.plot(df['Date'], df['kijun_sen'], label='Kijun-sen', color='red', alpha=0.7)
    plt.fill_between(df['Date'], df['senkou_span_a'], df['senkou_span_b'], color='lightgray', alpha=0.5)
    plt.title('Price, Ichimoku Cloud & Supertrend')
    plt.legend(loc='best')

    # Plotting Cumulative Return
    plt.subplot(3, 1, 2)
    plt.plot(df['Date'], df['cumulative_return'], label='Cumulative Strategy Return', color='purple', alpha=0.7)
    plt.title('Cumulative Return of Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend(loc='best')

    # Plotting Drawdown
    plt.subplot(3, 1, 3)
    plt.fill_between(df['Date'], df['drawdown'], color='red', alpha=0.3)
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url

# Example usage
if __name__ == "__main__":
    # Simulating some random data for testing
    df = pd.DataFrame({
        'Date': pd.date_range(start='1/1/2025', periods=100),
        'Close': np.random.rand(100) * 100,
        'High': np.random.rand(100) * 100 + 10,
        'Low': np.random.rand(100) * 100 - 10
    })

    backtest_results, _, _ = backtest_strategy(df)
    plot_url = plot_backtest(backtest_results)
    print(backtest_results.tail())
    print(f"Plot URL: data:image/png;base64,{plot_url}")