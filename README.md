# AgenticAI Crypto Coin Price Predictor

##AgenticAI is a comprehensive and reliable crypto coin price predictor that leverages advanced machine learning models to forecast the next day’s closing price of various cryptocurrencies. Alongside predictive analytics, AgenticAI provides robust risk management insights, technical indicators, backtesting capabilities, and dynamic report generation.

Key Features
	1.	##Data Fetching
	•	Fetch historical data from multiple sources, including Binance and Yahoo Finance (yfinance).
	•	Specify your desired cryptocurrency symbol (e.g., BTCUSDT, ETHUSDT, BTC-USD, etc.).
	2.	##Technical Analysis
	•	Add popular technical indicators such as SMA, EMA, RSI, MACD, Bollinger Bands, and more.
	•	Easily view the indicator data and incorporate it into backtesting or predictions.
	3.	##Price Prediction
	•	Predict next-day closing prices using an ensemble of machine learning models.
	•	Leverage advanced architectures, including LSTM, XGBoost, and ensemble stacking for more accurate forecasts.
	4.	##Backtesting
	•	Run backtests on built-in or custom trading strategies.
	•	View performance metrics such as Sharpe Ratio, Max Drawdown, and cumulative returns.
	5.	##Risk Management
	•	Get support/resistance levels, dynamic stop-loss/take-profit calculations, ATR (Average True Range), and recommended position sizing.
	•	Monitor your risk-to-reward ratios for more informed decision-making.
	6.	##Report Generation
	•	Generate comprehensive reports with visualizations, including Fibonacci retracement, candlestick charts, and risk analysis overlays.
	•	Export these reports in interactive HTML format for easy sharing.
	7.	##User-Friendly UI
	•	A dynamic table logs each action (e.g., Fetch Data, Predict, Backtest) with timestamps, details, and optional embedded charts.
	•	A global loader/spinner ensures smooth user experience while background tasks run.
	•	Color-coded circular buttons for quick access to each functionality.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AgenticAI_stock_predictor.git
   cd AgenticAI_stock_predictor
Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required dependencies:

pip install -r requirements.txt
Usage
Run the Flask App:

python app.py
Access the Web Interface: Open your web browser and go to http://127.0.0.1:5000/.

Fetch Data: Enter the symbol of the cryptocurrency (e.g., bitcoin) and click on "Fetch Data" to retrieve historical data.

Perform Technical Analysis: Click on "Technical Analysis" to add technical indicators to the fetched data.

Predict Price: Click on "Predict" to predict the next day's closing price using the ensemble model.

Backtest Strategy: Click on "Backtest" to backtest the trading strategy and visualize the results.

Risk Analysis: Click on "Risk Analysis" to get advanced risk management insights.

Generate Report: Click on "Generate Report" to generate a comprehensive report with visualizations.

Project Structure
app.py: Main Flask application file.
data_fetcher.py: Module for fetching historical data.
technical_analysis.py: Module for adding technical indicators.
ai_model.py: Module for training and predicting prices using machine learning models.
backtesting.py: Module for backtesting trading strategies.
risk_management.py: Module for providing risk management insights.
reports.py: Module for generating reports.
templates/index.html: HTML template for the web interface.
requirements.txt: List of required dependencies.

Screenshots:
![image](https://github.com/user-attachments/assets/b92a3856-3d45-40e8-aa53-a251254c9d9f)
![image](https://github.com/user-attachments/assets/d75a907b-f98f-40f2-86fc-2cac7ecc962e)
![image](https://github.com/user-attachments/assets/ebc176cb-7f64-4866-98e4-f23ee876bacc)
![image](https://github.com/user-attachments/assets/99c1da58-9f8d-4258-b7ce-d942a2fa5b77)
![image](https://github.com/user-attachments/assets/8407bdcc-c676-430e-89bf-15bebae84560)
![image](https://github.com/user-attachments/assets/09be8b8b-2b31-4f00-8598-a12b934f68bf)
![newplot (1)](https://github.com/user-attachments/assets/e3481427-b62c-4289-88b1-902b75312ca6)


Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
XGBoost
Scikit-Learn
Pandas
Flask
Matplotlib
mplfinance
Contact
For any questions or inquiries, please contact [chauhanamit090@hotmail.com].

Happy Predicting! 🚀
