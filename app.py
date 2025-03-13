import base64
import io
from datetime import datetime, timedelta
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory, render_template
import os
import pandas as pd
from matplotlib import pyplot as plt
from config import Config
from data_fetcher import fetch_historical_data
from technical_analysis import add_technical_indicators
from ai_model import predict_price, load_model
from risk_management import risk_analysis
from backtesting import backtest_strategy, plot_backtest
from reports import generate_report

app = Flask(__name__)
app.config.from_object(Config)
# Simple in-memory cache for fetched data (per symbol)
data_cache = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_data', methods=['GET'])
def fetch_data():
    symbol = request.args.get('symbol', 'bitcoin')
    days = request.args.get('days', '365')
    interval = request.args.get('interval', 'daily')
    try:
        df = fetch_historical_data(symbol, days, interval)
        data_cache[symbol] = df
        return jsonify({"message": f"Data fetched for {symbol}", "rows": len(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/technical_analysis', methods=['GET'])
def technical_analysis():
    symbol = request.args.get('symbol', 'bitcoin')
    if symbol not in data_cache:
        return jsonify({"error": "Data not fetched. Please call /fetch_data first."}), 400
    df = data_cache[symbol]
    try:
        df = add_technical_indicators(df)
        data_cache[symbol] = df  # Update cache with new indicators
        # Return the latest row of technical analysis
        return jsonify(df.tail(1).to_dict(orient="records")[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', 'bitcoin')
    model_type = request.args.get('model', 'xgboost')
    if symbol not in data_cache:
        return jsonify({"error": "Data not fetched. Please call /fetch_data first."}), 400
    df = data_cache[symbol]
    # Ensure technical indicators are added
    df = add_technical_indicators(df)
    if model_type.lower() == 'xgboost':
        result = predict_price(df)
    elif model_type.lower() == 'lstm':
        result = load_model(df)
    else:
        return jsonify({"error": "Invalid model type. Use 'xgboost' or 'lstm'."}), 400
    return jsonify(result)

@app.route('/backtest', methods=['GET'])
def backtest():
    symbol = request.args.get('symbol', 'bitcoin')
    if symbol not in data_cache:
        return jsonify({"error": "Data not fetched. Please call /fetch_data first."}), 400
    df = data_cache[symbol]
    try:
        backtest_results, sharpe_ratio, max_drawdown = backtest_strategy(df)
        # Plot the backtest results
        plot_url = plot_backtest(backtest_results)
        # Return the last 10 rows of backtesting results and the plot URL
        return jsonify({
            "backtest_results": backtest_results.tail(10).to_dict(orient="records"),
            "plot_url": f"data:image/png;base64,{plot_url}",
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/risk_analysis', methods=['GET'])
def risk():
    symbol = request.args.get('symbol', 'bitcoin')
    if symbol not in data_cache:
        return jsonify({"error": "Data not fetched. Please call /fetch_data first."}), 400
    df = data_cache[symbol]
    try:
        risk_info = risk_analysis(df)
        return jsonify(risk_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate_report', methods=['GET'])
def generate_report_route():
    symbol = request.args.get('symbol', '')

    if symbol:
        try:
            # Simulate fetching data for the symbol
            end_date = datetime.today()
            start_date = end_date - timedelta(days=365)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')

            # Simulate the data (replace this with real data fetching if necessary)
            data = {
                'Date': dates,
                'Open': np.random.uniform(1000, 2000, len(dates)),
                'High': np.random.uniform(2000, 2500, len(dates)),
                'Low': np.random.uniform(800, 1000, len(dates)),
                'Close': np.random.uniform(1500, 2000, len(dates)),
            }
            df = pd.DataFrame(data)

            # Generate the report (No filename argument here)
            result = generate_report(df)

            # If the result contains an error, log and return the error message
            if "error" in result:
                app.logger.error(f"Error generating report: {result['error']}")
                return jsonify(result), 400  # Return error message if report generation failed

            # Successfully generated the report, return the URL
            app.logger.info(f"Report successfully generated for symbol: {symbol}")
            return jsonify({"report_file": result["report_file"]})

        except Exception as e:
            # Catch all exceptions and log them
            app.logger.error(f"Error in generate_report: {str(e)}")
            return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500  # Return a detailed error response
    else:
        app.logger.error("No symbol provided")
        return jsonify({"error": "No symbol provided"}), 400


if __name__ == '__main__':
    app.run(debug=True)