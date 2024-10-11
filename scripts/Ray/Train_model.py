import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ray
import time  # Import time module

# Initialize Ray with a specific number of workers
ray.init()

# Load the dataset
data_path = '../../Data/dataset_RL.csv'
data = pd.read_csv(data_path)

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data[['Open', 'Close', 'High', 'Low']]  # Select features

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create input sequences
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length][[0, 1]])  # Predict Open and Close
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Use the last time step

# Model parameters
input_size = X.shape[2]
hidden_size = 50
output_size = 2  # Open and Close prices

model = LSTMModel(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
@ray.remote
def train_model(X_train, y_train):
    model.train()
    for epoch in range(100):  # Train for 100 epochs
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X_train))
        loss = criterion(outputs, torch.FloatTensor(y_train))
        loss.backward()
        optimizer.step()
    return model.state_dict()

# Measure training time
start_time = time.time()  # Start timer

# Train the model in parallel but limit parallelism
num_workers = 2  # Reduce parallelism to 2 workers to avoid memory overload
futures = [train_model.remote(X_train, y_train) for _ in range(num_workers)]  # Use limited workers
models_state = ray.get(futures)

# End time
end_time = time.time()  # End timer

# Print the total training time
print(f"Training Time: {end_time - start_time:.2f} seconds")

# Save the last trained model
torch.save(models_state[-1], 'stock_price_model_ray.pth')

ray.shutdown()  # Shut down Ray
