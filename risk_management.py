import pandas as pd
import numpy as np


# ----------------------- TECHNICAL INDICATORS -----------------------
def calculate_support_resistance(df, window=20):
    """Calculate support, resistance, and Fibonacci retracement levels."""
    df['Support'] = df['Close'].rolling(window=window).min()
    df['Resistance'] = df['Close'].rolling(window=window).max()

    # Fibonacci retracement levels
    max_price = df['Resistance'].iloc[-1]
    min_price = df['Support'].iloc[-1]

    df['Fib_23.6'] = max_price - 0.236 * (max_price - min_price)
    df['Fib_38.2'] = max_price - 0.382 * (max_price - min_price)
    df['Fib_50'] = max_price - 0.5 * (max_price - min_price)
    df['Fib_61.8'] = max_price - 0.618 * (max_price - min_price)

    return df


def calculate_atr(df, window=14):
    """Calculate the Average True Range (ATR)."""
    df['High-Low'] = df['High'] - df['Low']
    df['High-Close'] = (df['High'] - df['Close'].shift()).abs()
    df['Low-Close'] = (df['Low'] - df['Close'].shift()).abs()
    df['TrueRange'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=window).mean()
    return df


def calculate_ichimoku(df):
    """Calculate Ichimoku Cloud indicators."""
    df['Tenkan_Sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    df['Kijun_Sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
    df['Senkou_Span_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    df['Chikou_Span'] = df['Close'].shift(-26)
    return df


def calculate_supertrend(df, atr_multiplier=3, window=14):
    """Calculate the Supertrend indicator."""
    df = calculate_atr(df, window)

    # Upper and lower bands
    df['Upper_Band'] = (df['High'] + df['Low']) / 2 + atr_multiplier * df['ATR']
    df['Lower_Band'] = (df['High'] + df['Low']) / 2 - atr_multiplier * df['ATR']

    # Initialize Supertrend
    df['Supertrend'] = np.nan

    # Supertrend Calculation
    for i in range(1, len(df)):
        if df['Close'][i] <= df['Upper_Band'][i - 1]:
            df.loc[i, 'Supertrend'] = df['Upper_Band'][i]
        else:
            df.loc[i, 'Supertrend'] = df['Lower_Band'][i]

    return df


# ----------------------- RISK ANALYSIS FUNCTION -----------------------
def risk_analysis(df, account_balance=10000, risk_percentage=0.02):
    """
    Perform advanced risk analysis, including:
    - Support & resistance levels
    - ATR-based stop-loss/take-profit
    - Risk-to-reward ratio
    - Position sizing
    - Ichimoku & Supertrend signals
    """
    df = calculate_support_resistance(df)
    df = calculate_atr(df)
    df = calculate_ichimoku(df)
    df = calculate_supertrend(df)

    latest = df.iloc[-1]

    # Stop-loss and take-profit using ATR
    stop_loss_atr = latest['Close'] - (latest['ATR'] * 1.5)
    take_profit_atr = latest['Close'] + (latest['ATR'] * 2)

    # Risk-to-reward ratio
    risk = latest['Close'] - stop_loss_atr
    reward = take_profit_atr - latest['Close']
    risk_to_reward_ratio = reward / risk if risk != 0 else np.nan

    # Position sizing
    dollar_risk = account_balance * risk_percentage
    position_size = dollar_risk / abs(latest['Close'] - stop_loss_atr)

    # Ichimoku & Supertrend signals
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
