import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from risk_management import risk_analysis


def add_moving_averages(df):
    """
    Adds moving averages (SMA and EMA) to the dataframe.
    """
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    return df


def calculate_rsi(df, window=14):
    """
    Calculates the Relative Strength Index (RSI).
    """
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def plot_fibonacci(df):
    """
    Plots Fibonacci retracement levels based on the high and low of the price data.
    """
    high_price = df['High'].max()
    low_price = df['Low'].min()

    diff = high_price - low_price
    levels = {
        'Level 0%': high_price,
        'Level 23.6%': high_price - 0.236 * diff,
        'Level 38.2%': high_price - 0.382 * diff,
        'Level 50%': high_price - 0.5 * diff,
        'Level 61.8%': high_price - 0.618 * diff,
        'Level 100%': low_price,
    }

    return levels


def plot_interactive_chart(df, filename="interactive_report.html"):
    """
    Creates an interactive candlestick chart with Plotly, including key indicators and risk management levels.
    """
    # Add Moving Averages and RSI
    df = add_moving_averages(df)
    df = calculate_rsi(df)

    # Get the latest risk analysis data
    risk_data = risk_analysis(df)

    # Calculate Fibonacci levels
    fib_levels = plot_fibonacci(df)

    # Create subplots for the candlestick chart and RSI
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price and Technical Indicators', 'RSI (14)'),
        row_titles=['Price', 'RSI'],
        horizontal_spacing=0.1
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='Candlestick',
        increasing=dict(line=dict(color='#1E8449')),  # Green for up
        decreasing=dict(line=dict(color='#E74C3C')),  # Red for down
        showlegend=False
    ), row=1, col=1)

    # Moving Averages (SMA & EMA)
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='#3498DB', width=2)),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='#F39C12', width=2)),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['EMA_20'], mode='lines', name='EMA 20', line=dict(color='#8E44AD', width=2)),
        row=1, col=1)

    # Plot Fibonacci Levels
    for level_name, level_value in fib_levels.items():
        fig.add_trace(go.Scatter(
            x=df['Date'], y=[level_value] * len(df), mode='lines', name=level_name,
            line=dict(dash='dash', width=2)
        ), row=1, col=1)

    # RSI Indicator
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='#F39C12', width=2)),
                  row=2, col=1)
    fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=2, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=2, col=1)

    # Add support, resistance, stop-loss, and take-profit levels
    fig.add_trace(go.Scatter(
        x=df['Date'], y=[risk_data['support']] * len(df), mode='lines', name='Support',
        line=dict(color='#27AE60', dash='dot')),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=[risk_data['resistance']] * len(df), mode='lines', name='Resistance',
        line=dict(color='#E74C3C', dash='dot')),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=[risk_data['suggested_stop_loss']] * len(df), mode='lines', name='Stop Loss',
        line=dict(color='#8E44AD', dash='dot')),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=[risk_data['suggested_take_profit']] * len(df), mode='lines', name='Take Profit',
        line=dict(color='#2980B9', dash='dot')),
        row=1, col=1)

    # Customize layout
    fig.update_layout(
        title="Interactive Crypto Chart with Technical Indicators and Risk Management",
        xaxis_rangeslider_visible=False,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_tickformat='%Y-%m-%d',
        template="plotly_dark",
        height=800,
        showlegend=True
    )

    fig.write_html(filename)
    return filename


def generate_report(df, filename="full_report.html"):
    try:
        # Ensure the static directory exists
        static_dir = os.path.join(os.getcwd(), 'static')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        # Generate the interactive chart or report
        interactive_chart_file = plot_interactive_chart(df, filename=os.path.join(static_dir, filename))

        # Ensure the report file is generated correctly
        if not interactive_chart_file:
            raise Exception("Failed to generate the report.")

        return {"report_file": f"/static/{filename}"}
    except Exception as e:
        # Handle errors gracefully
        return {"error": str(e)}


# Example Usage
if __name__ == "__main__":
    # Generate date range for the last 1 year
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    # Create a dummy dataframe for testing with the new date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = {
        'Date': dates,
        'Open': np.random.uniform(1000, 2000, len(dates)),
        'High': np.random.uniform(2000, 2500, len(dates)),
        'Low': np.random.uniform(800, 1000, len(dates)),
        'Close': np.random.uniform(1500, 2000, len(dates)),
    }
    df = pd.DataFrame(data)

    # Generate report
    result = generate_report(df)
    print(f"Interactive report saved as: {result['report_file']}")
