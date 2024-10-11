import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time  # Import time module

# Define the device (CPU in this case)
device = torch.device('cpu')

# Load the dataset
data_path = '../../Data/dataset_RL.csv'
data = pd.read_csv(data_path)

# Load trained model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Load model
model = LSTMModel(input_size=4, hidden_size=50, output_size=2)
model.load_state_dict(torch.load('stock_price_model_ray.pth'))
model.eval()

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data[['Open', 'Close', 'High', 'Low']]  # Ensure this matches your training data features

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to make predictions for a given date
def predict_price(date_str):
    start_time = time.time()  # Start timer
    
    date = pd.to_datetime(date_str)
    past_data = data[data.index < date].tail(30)  # Last 30 days of data

    if len(past_data) < 30:
        print(f"Not enough data to predict for {date_str}.")
        return

    past_data_scaled = scaler.transform(past_data)
    X_input = torch.tensor(past_data_scaled, dtype=torch.float32).view(1, 30, -1).to(device)  # Reshape for LSTM

    # Predict the next Open and Close prices
    with torch.no_grad():
        predicted = model(X_input)

    # Get the predicted prices
    predicted_prices = predicted.cpu().numpy()[0]  # Convert to numpy array and get the first element

    # Create a dummy array for inverse_transform with the same shape as scaler's fitted data
    dummy_array = np.zeros((1, 4))  # Create an array with the same number of features
    dummy_array[0, 0] = predicted_prices[0]  # Assign predicted Open price
    dummy_array[0, 1] = predicted_prices[1]  # Assign predicted Close price

    # Inverse transform using the dummy array
    inverse_prices = scaler.inverse_transform(dummy_array)

    # Get actual prices for the specified date
    try:
        actual_price = data.loc[date_str]
        # Print actual Open and Close prices
        print(f"Actual Open price for {date_str}: {actual_price['Open']:.2f}")
        print(f"Actual Close price for {date_str}: {actual_price['Close']:.2f}")

        # Calculate accuracy only if actual prices are not zero
        if actual_price['Open'] != 0 and actual_price['Close'] != 0:
            open_accuracy = 100 - (abs(inverse_prices[0][0] - actual_price['Open']) / actual_price['Open'] * 100)
            close_accuracy = 100 - (abs(inverse_prices[0][1] - actual_price['Close']) / actual_price['Close'] * 100)

            print(f"Open price accuracy: {open_accuracy:.2f}%")
            print(f"Close price accuracy: {close_accuracy:.2f}%")
        else:
            print("Actual prices are zero, unable to calculate accuracy.")
    except KeyError:
        print(f"No actual prices available for the date: {date_str}.")

    end_time = time.time()  # End timer
    # Print the total execution time
    print(f"Execution Time: {end_time - start_time:.2f} seconds")

# Example usage
input_date = '2023-02-01'  # Change this date to any date in 2023
predict_price(input_date)
