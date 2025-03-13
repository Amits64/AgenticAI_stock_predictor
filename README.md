# AgenticAI Crypto Coin Price Predictor

AgenticAI is a sophisticated and reliable crypto coin price predictor that leverages advanced machine learning models to predict the next day's closing price of various cryptocurrencies. It also provides risk management insights, including dynamic stop-loss/take-profit levels, position sizing, and risk-to-reward ratios.

## Features

- **Data Fetching**: Fetch historical data for various cryptocurrencies.
- **Technical Analysis**: Add technical indicators such as SMA, EMA, RSI, MACD, and more.
- **Price Prediction**: Predict the next day's closing price using an ensemble of models.
- **Backtesting**: Backtest trading strategies and visualize the results.
- **Risk Management**: Provide advanced risk management insights, including support/resistance levels, ATR, and position sizing.
- **Report Generation**: Generate comprehensive reports with visualizations.

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
![image](https://github.com/user-attachments/assets/83021c75-6b73-4530-acb4-b6d8315c7df3)
![image](https://github.com/user-attachments/assets/9570302f-72a8-4a45-8b1b-f66d955bf4cf)
![image](https://github.com/user-attachments/assets/c04a2e1e-5ba3-4ff7-bf5e-e763e8c3364e)
![image](https://github.com/user-attachments/assets/638a3a84-8cf1-4f9d-983d-cecea32a1905)
![image](https://github.com/user-attachments/assets/84f0b294-9185-42bc-b8c0-79de22d06169)

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

