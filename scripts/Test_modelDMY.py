import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import time  # Import time module to measure prediction time
import mlflow
import mlflow.keras

# Set the tracking URI and experiment name
mlflow.set_tracking_uri("http://localhost:5000")  # Ensure the path is correct
mlflow.set_experiment("LSTM Prediction")

# Load the dataset
data_path = '../Data/dataset_RL.csv'
data = pd.read_csv(data_path)

# Load the trained model
model = load_model('stock_price_model_DMY.h5')

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Select the same features as used during training (e.g., Open, Close, High, Low)
features = ['Open', 'Close', 'High', 'Low']
data = data[features]  # Ensure this matches your training data features

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to make predictions for a given date
def predict_price(date_str):
    date = pd.to_datetime(date_str)
    
    # Start an MLflow run
    with mlflow.start_run():
        # Retrieve the past 30 days of data before the specified date
        past_data = data[data.index < date].tail(30)
        
        if len(past_data) < 30:  # Ensure you have enough data for prediction
            print(f"Not enough data to predict for {date_str}.")
            return
        
        # Scale the past data
        past_data_scaled = scaler.transform(past_data)
        
        # Reshape the data for the LSTM model (1, time_steps, num_features)
        X_input = past_data_scaled.reshape(1, past_data_scaled.shape[0], past_data_scaled.shape[1])
        
        # Record the start time for prediction
        start_time = time.time()
        
        # Predict the next Open and Close prices
        predicted = model.predict(X_input)
        
        # Record the end time for prediction
        end_time = time.time()
        
        # Calculate and print the prediction time
        prediction_time = end_time - start_time
        print(f"Prediction time: {prediction_time} seconds")
        
        # The model likely predicts in normalized space, so we inverse the scaling
        predicted_prices_scaled = np.zeros((1, len(features)))
        predicted_prices_scaled[0, 0:2] = predicted[0]
        
        # Inverse transform to get the actual predicted prices
        predicted_prices = scaler.inverse_transform(predicted_prices_scaled)[0]
        
        # Retrieve actual prices for the specified date
        try:
            actual_price = data.loc[date_str][['Open', 'Close']]
        except KeyError:
            print(f"No data available for the date {date_str}.")
            return
        
        # Print actual Open and Close prices
        print(f"Actual Open price for {date_str}: {actual_price['Open']:.2f}")
        print(f"Actual Close price for {date_str}: {actual_price['Close']:.2f}")

        # Calculate accuracy
        open_accuracy = 100 - (abs(predicted_prices[0] - actual_price['Open']) / actual_price['Open'] * 100)
        close_accuracy = 100 - (abs(predicted_prices[1] - actual_price['Close']) / actual_price['Close'] * 100)

        # Print predicted prices and accuracy
        print(f"Predicted Open price for {date_str}: {predicted_prices[0]:.2f}")
        print(f"Predicted Close price for {date_str}: {predicted_prices[1]:.2f}")
        print(f"Open price accuracy: {open_accuracy:.2f}%")
        print(f"Close price accuracy: {close_accuracy:.2f}%")

        # Log parameters and metrics to MLflow
        mlflow.log_param("input_date", date_str)
        mlflow.log_param("model", "stock_price_model_DMY.h5")
        mlflow.log_metric("predicted_open", predicted_prices[0])
        mlflow.log_metric("predicted_close", predicted_prices[1])
        mlflow.log_metric("actual_open", actual_price['Open'])
        mlflow.log_metric("actual_close", actual_price['Close'])
        mlflow.log_metric("open_accuracy", open_accuracy)
        mlflow.log_metric("close_accuracy", close_accuracy)
        mlflow.log_metric("prediction_time", prediction_time)

# Example usage
predict_price('2023-01-03')
