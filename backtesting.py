#backtesting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64


def moving_average_crossover_strategy(df, short_window=20, long_window=50):
    """
    Implements a simple moving average crossover strategy.
    """
    df = df.copy()

    # Handle edge cases where there are not enough data points for rolling windows
    df['short_ma'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
    df['long_ma'] = df['Close'].rolling(window=long_window, min_periods=1).mean()

    # Calculate the signal when the short MA crosses above the long MA (1) or below (0)
    df['signal'] = 0
    df.loc[short_window:, 'signal'] = (df['short_ma'][short_window:] > df['long_ma'][short_window:]).astype(int)

    # Capture the position changes (signals 1 for buy, -1 for sell)
    df['position'] = df['signal'].diff()

    return df


def backtest_strategy(df, short_window=20, long_window=50, initial_capital=10000, position_size=1, slippage=0.0001):
    """
    Backtests the moving average crossover strategy.
    Calculates daily returns when in position and computes cumulative returns.
    """
    df = moving_average_crossover_strategy(df, short_window, long_window)

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

    return df[
        ['Date', 'Close', 'short_ma', 'long_ma', 'signal', 'cumulative_return', 'portfolio_value', 'position_size',
         'slippage_adjusted', 'drawdown']], sharpe_ratio, max_drawdown


def plot_backtest(df):
    """
    Plots the results of the backtest, including moving averages, equity curve, and drawdowns.
    """
    plt.figure(figsize=(14, 10))

    # Plotting Closing Price and Moving Averages
    plt.subplot(3, 1, 1)
    plt.plot(df['Date'], df['Close'], label='Closing Price', color='blue', alpha=0.7)
    plt.plot(df['Date'], df['short_ma'], label='Short MA (20)', color='green', alpha=0.7)
    plt.plot(df['Date'], df['long_ma'], label='Long MA (50)', color='red', alpha=0.7)
    plt.title('Price and Moving Averages')
    plt.legend(loc='best')
    plt.xlabel('Date')
    plt.ylabel('Price')

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


def optimize_parameters(df, short_window_range=(10, 30), long_window_range=(30, 100)):
    """
    Optimizes the moving average crossover strategy by testing different parameter ranges.
    """
    best_sharpe = -np.inf
    best_params = None
    best_df = None

    # Loop through all combinations of short and long window parameters
    for short_window in range(short_window_range[0], short_window_range[1] + 1, 5):
        for long_window in range(long_window_range[0], long_window_range[1] + 1, 10):
            print(f"Testing short_window={short_window}, long_window={long_window}")
            df_temp, sharpe_ratio, _ = backtest_strategy(df, short_window=short_window, long_window=long_window)

            # Calculate Sharpe ratio for optimization
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_params = (short_window, long_window)
                best_df = df_temp

    print(f"Best Parameters: short_window={best_params[0]}, long_window={best_params[1]}")
    return best_df


# Example usage
if __name__ == "__main__":
    # Assuming you have a DataFrame `df` with the necessary columns
    df = pd.DataFrame({
        'Date': pd.date_range(start='1/1/2020', periods=100),
        'Close': np.random.rand(100) * 100
    })
    backtest_results, _, _ = backtest_strategy(df)
    plot_url = plot_backtest(backtest_results)
    print(backtest_results.tail())
    print(f"Plot URL: data:image/png;base64,{plot_url}")