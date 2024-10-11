import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load the dataset
data_path = '../Data/dataset_RL.csv'
data = pd.read_csv(data_path)

# Load the trained model
model = load_model('stock_price_model.h5')

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Use only the 'Open' and 'Close' prices
data = data[['Open', 'Close']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to make predictions for a given date
def predict_price(date_str):
    date = pd.to_datetime(date_str)
    past_data = data[data.index < date].tail(7)  # Get last 7 days before the specified date
    if len(past_data) < 7:
        print(f"Not enough data to predict for {date_str}.")
        return
    
    past_data_scaled = scaler.transform(past_data)
    X_input = past_data_scaled.reshape(1, past_data_scaled.shape[0], 2)
    
    # Predict the next Open and Close prices
    predicted = model.predict(X_input)
    predicted_prices = scaler.inverse_transform(predicted)[0]
    
    # Get actual prices for the specified date
    actual_price = data.loc[date_str]

    # Print actual Open and Close prices
    print(f"Actual Open price for {date_str}: {actual_price['Open']:.2f}")
    print(f"Actual Close price for {date_str}: {actual_price['Close']:.2f}")

    # Calculate accuracy
    open_accuracy = 100 - (abs(predicted_prices[0] - actual_price['Open']) / actual_price['Open'] * 100)
    close_accuracy = 100 - (abs(predicted_prices[1] - actual_price['Close']) / actual_price['Close'] * 100)

    print(f"Predicted Open price for {date_str}: {predicted_prices[0]:.2f}")
    print(f"Predicted Close price for {date_str}: {predicted_prices[1]:.2f}")
    print(f"Open price accuracy: {open_accuracy:.2f}%")
    print(f"Close price accuracy: {close_accuracy:.2f}%")

# Example usage
input_date = '2023-01-31'  # Change this date to any date in 2023
predict_price(input_date)
