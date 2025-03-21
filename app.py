from flask import Flask, request, jsonify, render_template
from config import Config
from data_fetcher import BinanceHistoricalData
from technical_analysis import TechnicalAnalysis
from ai_model import ModelManager
from risk_management import risk_analysis
from backtesting import backtest_strategy
from reports import generate_report
import os
import importlib

app = Flask(__name__)
app.config.from_object(Config)

# In-memory cache for fetched data (per symbol)
data_cache = {}

# Instantiate the model manager globally
model_manager = ModelManager()

# ---------------------------
# 🔥 Home Route
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

# ---------------------------
# 📊 Data Fetching Route
# ---------------------------
@app.route('/fetch_data', methods=['GET'])
def fetch_data():
    symbol = request.args.get('symbol', 'BTCUSDT')
    days = int(request.args.get('days', '365'))
    interval = request.args.get('interval', '1d')
    valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '3h', '6h', '12h', '1d', '3d', '1w', '1M']
    if interval not in valid_intervals:
        return jsonify({"error": f"Invalid interval. Valid intervals are: {', '.join(valid_intervals)}."}), 400
    try:
        fetcher = BinanceHistoricalData(
            Config.BINANCE_API_KEY,
            Config.BINANCE_API_SECRET,
            symbol=symbol,
            days=days,
            interval=interval
        )
        df = fetcher.fetch_historical_data()
        if df.empty:
            return jsonify({"error": "No data returned from Binance."}), 500
        data_cache[symbol] = df
        fetch_url = fetcher.generate_plot_url()
        return jsonify({
            "message": f"Data fetched successfully for {symbol}",
            "rows": len(df),
            "fetch_url": fetch_url
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# 📉 Technical Analysis Route
# ---------------------------
@app.route('/technical_analysis', methods=['GET'])
def technical_analysis():
    symbol = request.args.get('symbol', 'BTCUSDT')
    if symbol not in data_cache:
        return jsonify({"error": "Data not fetched. Please call /fetch_data first."}), 400
    try:
        ta = TechnicalAnalysis(csv_file="raw_data.csv", window_size=50)
        ta.add_indicators()
        ta.scale_and_window()
        ta.save_results()
        return jsonify({
            "message": f"Technical analysis applied successfully for {symbol}",
            "indicators": ta.df.tail(1).to_dict(orient="records")[0]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# 🤖 AI Prediction Route
# ---------------------------
@app.route('/predict', methods=['GET'])
def predict():
    try:
        results = model_manager.predict()
        return jsonify({
            "ensemble_prediction": float(results.get('ensemble_prediction', 0.0)),
            "lstm_prediction": float(results.get('lstm_prediction', 0.0)),
            "mse": float(results.get('mse', 0.0)),
            "predict_url": results.get('predict_url', "")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# 📑 Strategy Listing Route
# ---------------------------
@app.route('/strategies', methods=['GET'])
def list_strategies():
    try:
        strategy_dir = os.path.join(os.getcwd(), "strategy")
        if not os.path.exists(strategy_dir):
            return jsonify({"error": "Strategy directory not found."}), 404
        # List only Python files and remove the .py extension
        strategies = [os.path.splitext(f)[0] for f in os.listdir(strategy_dir) if f.endswith('.py')]
        return jsonify({"strategies": strategies})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------
# 📈 Backtesting Route with Strategy Selection
# ---------------------------
@app.route('/backtest', methods=['GET'])
def backtest():
    symbol = request.args.get('symbol', 'BTCUSDT')
    strategy_name = request.args.get('strategy', '')  # Strategy selection from directory

    if symbol not in data_cache:
        return jsonify({"error": "Data not fetched. Please call /fetch_data first."}), 400

    df = data_cache[symbol]

    try:
        if strategy_name:
            # Replace '+' with '_' to form a valid module name
            module_name = strategy_name.replace('+', '_')
            strategy_module = importlib.import_module(f"strategy.{module_name}")

            # Ensure the backtest_strategy function exists
            if hasattr(strategy_module, 'backtest_strategy'):
                backtest_func = strategy_module.backtest_strategy
            else:
                return jsonify(
                    {"error": f"Strategy '{strategy_name}' does not have 'backtest_strategy' function."}), 400

            # Execute the strategy's backtest function
            backtest_results, sharpe_ratio, max_drawdown, full_plot_url = backtest_func(df)

        else:
            # Use default backtesting function if no strategy specified
            backtest_results, sharpe_ratio, max_drawdown, full_plot_url = backtest_strategy(df)

        return jsonify({
            "backtest_results": backtest_results.tail(10).to_dict(orient="records"),
            "plot_url": full_plot_url,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        })

    except ModuleNotFoundError:
        return jsonify({"error": f"Strategy '{strategy_name}' not found."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# 🚀 Risk Analysis Route
# ---------------------------
@app.route('/risk_analysis', methods=['GET'])
def risk_analysis_route():
    symbol = request.args.get('symbol', 'BTCUSDT')
    account_balance = float(request.args.get('account_balance', 10000))
    risk_percentage = float(request.args.get('risk_percentage', 0.02))
    if symbol not in data_cache:
        return jsonify({"error": "Data not fetched. Please call /fetch_data first."}), 400
    df = data_cache[symbol]
    try:
        analysis_results = risk_analysis(df, account_balance, risk_percentage)
        return jsonify({
            "symbol": symbol,
            "account_balance": account_balance,
            "risk_percentage": risk_percentage,
            "risk_analysis": analysis_results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# 📑 Report Generation Route
# ---------------------------
@app.route('/generate_report', methods=['GET'])
def generate_report_route():
    symbol = request.args.get('symbol', '')
    if symbol not in data_cache:
        return jsonify({"error": "Data not fetched. Please call /fetch_data first."}), 400
    try:
        df = data_cache[symbol]
        result = generate_report(df)
        if "error" in result:
            app.logger.error(f"Error generating report: {result['error']}")
            return jsonify(result), 400
        app.logger.info(f"Report successfully generated for symbol: {symbol}")
        return jsonify({"report_file": result["report_file"]})
    except Exception as e:
        app.logger.error(f"Error in generate_report: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ---------------------------
# 🚀 Health Check Route
# ---------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running", "message": "API is healthy"}), 200

# ---------------------------
# 🚀 Run Flask App
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)
