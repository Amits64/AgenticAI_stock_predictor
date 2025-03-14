import pandas as pd
import numpy as np

def calculate_support_resistance(df, window=20):
    """
    Calculate support and resistance levels as rolling minimum and maximum.
    Additionally, includes more advanced support/resistance levels based on Fibonacci retracements.
    """
    df['Support'] = df['Close'].rolling(window=window).min()
    df['Resistance'] = df['Close'].rolling(window=window).max()

    # Adding Fibonacci retracement levels (an additional technique for support/resistance)
    max_price = df['Resistance'].iloc[-1]
    min_price = df['Support'].iloc[-1]

    # Fibonacci levels are commonly used at 23.6%, 38.2%, 50%, 61.8% retracement levels.
    df['Fib_23.6'] = max_price - 0.236 * (max_price - min_price)
    df['Fib_38.2'] = max_price - 0.382 * (max_price - min_price)
    df['Fib_50'] = max_price - 0.5 * (max_price - min_price)
    df['Fib_61.8'] = max_price - 0.618 * (max_price - min_price)

    return df

def calculate_atr(df, window=14):
    """
    Calculate the Average True Range (ATR), which is useful for setting dynamic stop-loss/take-profit.
    """
    df['High-Low'] = df['High'] - df['Low']
    df['High-Close'] = (df['High'] - df['Close'].shift()).abs()
    df['Low-Close'] = (df['Low'] - df['Close'].shift()).abs()
    df['TrueRange'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=window).mean()
    return df

def calculate_ichimoku(df):
    """
    Calculate the Ichimoku Cloud indicators: Tenkan-Sen, Kijun-Sen, Senkou Span A, Senkou Span B.
    """
    # Calculate the 9-period high and low (for Tenkan-Sen)
    df['Tenkan_Sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2

    # Calculate the 26-period high and low (for Kijun-Sen)
    df['Kijun_Sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2

    # Senkou Span A: (Tenkan-Sen + Kijun-Sen) / 2, shifted forward 26 periods
    df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)

    # Senkou Span B: 52-period high + 52-period low / 2, shifted forward 26 periods
    df['Senkou_Span_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)

    # Chikou Span: Close price shifted backward 26 periods
    df['Chikou_Span'] = df['Close'].shift(-26)

    return df

def calculate_supertrend(df, atr_multiplier=3, window=14):
    """
    Calculate the Supertrend indicator.
    """
    df = calculate_atr(df, window)  # Ensure ATR is calculated first

    # Calculate the basic upper and lower bands
    df['Upper_Band'] = (df['High'] + df['Low']) / 2 + atr_multiplier * df['ATR']
    df['Lower_Band'] = (df['High'] + df['Low']) / 2 - atr_multiplier * df['ATR']

    # Initialize the Supertrend column
    df['Supertrend'] = np.nan

    # Loop through the dataframe to compute Supertrend values
    for i in range(1, len(df)):
        if df['Close'][i] <= df['Upper_Band'][i - 1]:
            df.loc[i, 'Supertrend'] = df['Upper_Band'][i]
        else:
            df.loc[i, 'Supertrend'] = df['Lower_Band'][i]

    return df

def risk_analysis(df, account_balance=10000, risk_percentage=0.02):
    """
    Provides advanced risk management insights including dynamic stop-loss/take-profit levels,
    position sizing, and risk-to-reward ratio.
    Incorporates Ichimoku Cloud and Supertrend.
    """
    # Calculate support, resistance, ATR, Ichimoku Cloud, and Supertrend
    df = calculate_support_resistance(df)
    df = calculate_atr(df)
    df = calculate_ichimoku(df)
    df = calculate_supertrend(df)

    latest = df.iloc[-1]

    # Calculate Stop-Loss and Take-Profit using ATR (for volatility-adjusted levels)
    stop_loss_atr = latest['Close'] - (latest['ATR'] * 1.5)  # Using 1.5x ATR as stop loss
    take_profit_atr = latest['Close'] + (latest['ATR'] * 2)  # Using 2x ATR as take profit

    # Calculate the risk-to-reward ratio
    risk = latest['Close'] - stop_loss_atr
    reward = take_profit_atr - latest['Close']

    if risk == 0:
        risk_to_reward_ratio = np.nan  # Avoid division by zero
    else:
        risk_to_reward_ratio = reward / risk

    # Position Sizing: How much of the portfolio to risk based on the risk percentage
    dollar_risk = account_balance * risk_percentage
    position_size = dollar_risk / abs(latest['Close'] - stop_loss_atr)  # How many units of the asset to trade

    # Risk management decisions based on Ichimoku and Supertrend
    ichimoku_signal = "Bullish" if latest['Close'] > latest['Senkou_Span_A'] and latest['Close'] > latest['Senkou_Span_B'] else "Bearish"
    supertrend_signal = "Buy" if latest['Close'] > latest['Supertrend'] else "Sell"

    return {
        "current_close": latest['Close'],
        "support": latest['Support'],
        "resistance": latest['Resistance'],
        "fib_23.6": latest['Fib_23.6'],
        "fib_38.2": latest['Fib_38.2'],
        "fib_50": latest['Fib_50'],
        "fib_61.8": latest['Fib_61.8'],
        "suggested_stop_loss": stop_loss_atr,
        "suggested_take_profit": take_profit_atr,
        "risk_to_reward_ratio": risk_to_reward_ratio,
        "position_size": position_size,
        "account_balance_risked": dollar_risk,
        "ichimoku_signal": ichimoku_signal,
        "supertrend_signal": supertrend_signal
    }

# Example usage:
if __name__ == "__main__":
    # Get the current date and create a date range for the last year (365 days)
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=1)

    # Create the DataFrame with the date range from the last year
    df = pd.DataFrame({
        'Date': pd.date_range(start=start_date, end=end_date, freq='D'),
        'Open': np.random.rand(365) * 100 + 100,
        'High': np.random.rand(365) * 100 + 110,
        'Low': np.random.rand(365) * 100 + 90,
        'Close': np.random.rand(365) * 100 + 100
    })

    # Run risk analysis for a sample account balance of $10,000 and risk percentage of 2%
    result = risk_analysis(df, account_balance=10000, risk_percentage=0.02)
    print(result)