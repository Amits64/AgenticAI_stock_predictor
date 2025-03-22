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
![image](https://github.com/user-attachments/assets/8b1a172c-2612-4b82-b517-3dafd62bd51e)
![image](https://github.com/user-attachments/assets/b67d0692-3c66-4189-9bd2-655b2fce568d)
![image](https://github.com/user-attachments/assets/07af15a4-c281-4eb6-9d81-3ce500f0bccc)
![image](https://github.com/user-attachments/assets/4499f29a-a2fa-42fa-bbb5-4b98565f95d4)
![image](https://github.com/user-attachments/assets/4bc3717f-0b53-40e2-9182-106648428432)
![image](https://github.com/user-attachments/assets/a9b5dcb6-0590-4c44-a313-53b027b65724)
![image](https://github.com/user-attachments/assets/cea79300-5ee5-4ca6-820b-f221ff3fc6c9)
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
