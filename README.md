⸻

AgenticAI Crypto Coin Price Predictor

AgenticAI is a sophisticated and reliable cryptocurrency price predictor that leverages advanced machine learning models to forecast the next day’s closing price of various cryptocurrencies. It also provides in-depth risk management insights, including dynamic stop-loss/take-profit levels, position sizing, and risk-to-reward ratios.

⸻

Features

✅ Data Fetching: Fetch historical cryptocurrency data using APIs.
✅ Technical Analysis: Apply key indicators such as SMA, EMA, RSI, MACD, Bollinger Bands, ATR, and more.
✅ AI-Powered Price Prediction: Predict the next day’s closing price using an ensemble of machine learning models.
✅ Backtesting: Backtest trading strategies and visualize performance metrics.
✅ Risk Management: Get detailed insights, including support/resistance levels, ATR-based stop-loss, and dynamic position sizing.
✅ Comprehensive Report Generation: Generate interactive reports with visual analytics.

⸻

Installation

1. Clone the Repository

git clone https://github.com/yourusername/AgenticAI_stock_predictor.git
cd AgenticAI_stock_predictor

2. Create a Virtual Environment and Activate It

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install Required Dependencies

pip install -r requirements.txt



⸻

Usage

Step 1: Train the AI Model (Must Be Done First)

Before making predictions, you must train the AI model by executing the following command:

python train_model.py

This step generates the machine learning model that will be used by other scripts for price prediction.

Step 2: Run the Flask Web App

Once the model is trained, start the Flask application:

python app.py

Now, open your web browser and visit: http://127.0.0.1:5000/

Step 3: Fetch Historical Data
	•	Enter the cryptocurrency symbol (e.g., bitcoin, ethereum) and click on “Fetch Data” to retrieve historical price data.

Step 4: Perform Technical Analysis
	•	Click on “Technical Analysis” to apply key indicators to the fetched data.

Step 5: Predict the Next Day’s Price
	•	Click on “Predict” to use the trained AI model and forecast the next day’s closing price.

Step 6: Backtest Trading Strategies
	•	Click on “Backtest” to evaluate trading strategies and visualize their past performance.

Step 7: Risk Analysis & Insights
	•	Click on “Risk Analysis” to generate stop-loss levels, position sizing, and risk-to-reward calculations.

Step 8: Generate a Full Report
	•	Click on “Generate Report” to export a complete visual analysis of predictions, strategies, and insights.

⸻

Project Structure

AgenticAI_stock_predictor/
│── app.py                 # Main Flask application file
│── train_model.py         # Script to train the AI model (must run first)
│── data_fetcher.py        # Module for fetching historical data
│── technical_analysis.py  # Module for adding technical indicators
│── ai_model.py            # Core AI model for price prediction
│── backtesting.py         # Backtesting module for trading strategies
│── risk_management.py     # Module for risk analysis and position sizing
│── reports.py             # Generates detailed reports with visualizations
│── templates/index.html   # HTML template for the web interface
│── requirements.txt       # List of required dependencies
└── README.md              # Documentation



⸻

Screenshots


⸻

Contributing

Contributions are welcome! If you have ideas for improvements or bug fixes, feel free to open an issue or submit a pull request.

⸻

License

This project is licensed under the MIT License. See the LICENSE file for details.

⸻

Acknowledgments
	•	XGBoost
	•	Scikit-Learn
	•	Pandas
	•	Flask
	•	Matplotlib
	•	mplfinance

⸻

Contact

For any questions or inquiries, please reach out to: chauhanamit090@hotmail.com

Happy Predicting! 🚀

