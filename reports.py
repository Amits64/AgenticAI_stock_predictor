#reports.py
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from risk_management import risk_analysis


def generate_candlestick_chart(df, filename="report.png"):
    """
    Generates a static candlestick chart with technical indicators.
    """
    try:
        from mplfinance.original_flavor import candlestick_ohlc
    except ImportError:
        raise ImportError("Please install mplfinance to generate candlestick charts.")

    df_chart = df.copy()
    # Convert Date column to numerical format for plotting
    df_chart['Date_num'] = df_chart['Date'].map(pd.Timestamp.toordinal)
    ohlc = df_chart[['Date_num', 'Open', 'High', 'Low', 'Close']].values

    fig, ax = plt.subplots(figsize=(12, 6))
    candlestick_ohlc(ax, ohlc, width=0.6, colorup='g', colordown='r', alpha=0.8)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    ax.set_title("Candlestick Chart")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    return filename

def add_moving_averages(df):
    """
    Adds moving averages (SMA and EMA) to the dataframe.
    """
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
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


def plot_interactive_chart(df, filename="interactive_report.html"):
    """
    Creates an interactive candlestick chart with Plotly, including risk management levels.
    """
    df = add_moving_averages(df)
    df = calculate_rsi(df)

    # Get the latest risk analysis data
    risk_data = risk_analysis(df)

    # Create subplots for the candlestick chart and RSI indicator
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Candlestick Chart', 'RSI (14)'),
        row_titles=['Price', 'RSI'],
        horizontal_spacing=0.1
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='Candlestick',
        increasing=dict(line=dict(color='green')),  # Correct way to set increasing line color
        decreasing=dict(line=dict(color='red')),   # Correct way to set decreasing line color
        showlegend=False
    ), row=1, col=1)

    # Moving Averages
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='blue', width=1.5)),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', width=1.5)),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['EMA_20'], mode='lines', name='EMA 20', line=dict(color='purple', width=1.5)),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['EMA_50'], mode='lines', name='EMA 50', line=dict(color='brown', width=1.5)),
        row=1, col=1)

    # RSI Indicator
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='orange', width=1.5)),
                  row=2, col=1)
    fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=2, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=2, col=1)

    # Add support, resistance, stop-loss, and take-profit levels
    fig.add_trace(go.Scatter(
        x=df['Date'], y=[risk_data['support']] * len(df), mode='lines', name='Support', line=dict(color='green', dash='dot')),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=[risk_data['resistance']] * len(df), mode='lines', name='Resistance', line=dict(color='red', dash='dot')),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=[risk_data['suggested_stop_loss']] * len(df), mode='lines', name='Stop Loss', line=dict(color='purple', dash='dot')),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=[risk_data['suggested_take_profit']] * len(df), mode='lines', name='Take Profit', line=dict(color='blue', dash='dot')),
        row=1, col=1)

    # Customize layout
    fig.update_layout(
        title="Interactive Crypto Chart with Technical Indicators and Risk Management",
        xaxis_rangeslider_visible=False,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_tickformat='%Y-%m-%d',
        template="plotly_dark",
        height=800
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
