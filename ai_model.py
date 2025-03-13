# ai_model.py
import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import joblib
import numpy as np


def prepare_features(df: pd.DataFrame):
    """
    Prepare features for prediction.
    Uses 'Close', 'SMA_20', 'EMA_20', 'RSI', 'MACD', and additional features if available.
    """
    required_columns = ['Close', 'SMA_20', 'EMA_20', 'RSI', 'MACD_12_26_9']
    optional_columns = ['Volume']

    # Ensure necessary columns exist; drop rows with NaNs for simplicity.
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame.")

    features = df[required_columns].copy()

    # Add optional columns if they exist
    for col in optional_columns:
        if col in df.columns:
            features[col] = df[col]

    features = features.dropna()
    return features


def predict_price(df: pd.DataFrame, model=None):
    """
    Predicts the next day's closing price using an ensemble of models.
    """
    features = prepare_features(df)
    if features.empty:
        return {"message": "Not enough data to predict."}

    # Create target: next day closing price
    features = features.copy()
    features['Target'] = features['Close'].shift(-1)
    features = features.dropna()
    X = features.drop('Target', axis=1)
    y = features['Target']

    # Remove duplicate columns
    X = X.loc[:, ~X.columns.duplicated()]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split train/test without shuffling (time series)
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Use pre-trained model if provided
    if model is None:
        model = train_model(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    # Predict next day price using the last available row
    last_row = X_scaled[[-1]]
    next_day_pred = model.predict(last_row)[0]

    # Calculate the time to reach the predicted price
    time_to_reach_predicted_price = calculate_time_to_reach(df, next_day_pred)

    # Convert numpy.float32 to native Python float
    next_day_pred = float(next_day_pred)
    mse = float(mse)

    return {
        "next_day_prediction": next_day_pred,
        "test_mse": mse,
        "time_to_reach_predicted_price": time_to_reach_predicted_price
    }


def calculate_time_to_reach(df, predicted_price):
    """
    Calculate the time to reach the predicted price based on historical data.
    """
    current_price = df['Close'].iloc[-1]

    # Calculate the average daily return
    df['daily_return'] = df['Close'].pct_change()

    # Calculate the average daily return and standard deviation of returns
    avg_daily_return = df['daily_return'].mean()
    std_daily_return = df['daily_return'].std()

    # Calculate the number of days to reach the predicted price using a simple model
    if avg_daily_return != 0:
        days_to_reach = np.log(predicted_price / current_price) / avg_daily_return
        days_to_reach_stddev = np.log(predicted_price / current_price) / std_daily_return

        return {
            "average_days": days_to_reach,
            "stddev_days": days_to_reach_stddev
        }

    return {
        "average_days": float('inf'),
        "stddev_days": float('inf')
    }


def train_model(X_train, y_train):
    """
    Train the stacking model and return the trained model.
    """
    # Ensemble model
    estimators = [
        ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1)),
        ('ridge', Ridge(alpha=1.0)),
        ('knn', KNeighborsRegressor(n_neighbors=5)),
        ('svr', SVR(kernel='rbf'))
    ]
    model = StackingRegressor(estimators=estimators, final_estimator=XGBRegressor(n_estimators=100, learning_rate=0.1))
    model.fit(X_train, y_train)
    return model


def save_model(model, model_filename='ensemble_model.pkl'):
    """
    Save the trained model to a file.
    """
    try:
        # Check if the model is valid
        if model:
            # Ensure the file path exists
            model_filename = os.path.abspath(model_filename)
            joblib.dump(model, model_filename)
            print(f"Model saved to {model_filename}")
        else:
            print("Model is not valid, cannot save.")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")


def load_model(model_filename='ensemble_model.pkl'):
    """
    Load a pre-trained model from a file.
    """
    try:
        model_filename = os.path.abspath(model_filename)
        model = joblib.load(model_filename)
        print(f"Model loaded from {model_filename}")
        return model
    except FileNotFoundError:
        print(f"Model file {model_filename} not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Get the current date and create a date range for the last year (365 days)
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(days=364)

    # Create the DataFrame with the date range from the last year
    df = pd.DataFrame({
        'Date': pd.date_range(start=start_date, end=end_date, freq='D'),
        'Close': pd.Series(range(365)),
        'SMA_20': pd.Series(range(365)),
        'EMA_20': pd.Series(range(365)),
        'RSI': pd.Series(range(365)),
        'MACD_12_26_9': pd.Series(range(365))
    })

    # Load or train the model
    model = load_model('ensemble_model.pkl')
    if model is None:
        features = prepare_features(df)
        X = features.drop('Target', axis=1)
        y = features['Target']
        model = train_model(X, y)
        save_model(model, 'ensemble_model.pkl')

    # Predict price with the model
    result = predict_price(df, model)
    print(result)