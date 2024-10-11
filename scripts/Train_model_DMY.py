import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import time
import mlflow
import mlflow.keras  # Import mlflow.keras to log Keras models
from mlflow.models.signature import infer_signature  # To log model signature

# Set up MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")  # Ensure this path is correct
mlflow.set_experiment("LSTM Training")

# Load the dataset
data_path = '../Data/dataset_RL.csv'
data = pd.read_csv(data_path)

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Feature Engineering: Add cyclical features for month and day of year
data['Month'] = data.index.month
data['Day_of_Year'] = data.index.dayofyear

# Normalize the data (Open, Close, Month, Day_of_Year)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Open', 'Close', 'Month', 'Day_of_Year']])

# Prepare the data for training with a wider window to capture both short- and long-term patterns
def create_dataset(data, time_step=30):  # Use 30 days (monthly pattern) instead of 7 days
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])  # Use past 30 days of data
        y.append(data[i + time_step, :2])  # Predict only 'Open' and 'Close'
    return np.array(X), np.array(y)

# Set time_step for capturing 30-day trends (monthly) and train on 7 years of data (2015-2022)
time_step = 30
X, y = create_dataset(scaled_data, time_step)

# Split into training and testing sets (80% for training, 20% for testing)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape the data to fit the LSTM model (number of samples, time steps, number of features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 4)  # 4 features (Open, Close, Month, Day_of_Year)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 4)

# Build the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 4)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2))  # Predicting 2 values: 'Open' and 'Close' prices

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Record the start time
start_time = time.time()

# Start MLflow run
with mlflow.start_run() as run:
    # Log dataset as an artifact
    mlflow.log_artifact(data_path, artifact_path="dataset")

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Log metrics and parameters
    for epoch in range(10):
        mlflow.log_metric("loss", history.history['loss'][epoch], step=epoch)
    mlflow.log_param("epochs", 100)
    mlflow.log_param("batch_size", 32)

    # Record training time
    training_time = time.time() - start_time
    mlflow.log_metric("training_time", training_time)

    # Save the trained model and log with signature
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.keras.log_model(model, "stock_price_model", signature=signature)
    
    # Register the model
    mlflow.register_model(f"runs:/{run.info.run_id}/stock_price_model", "StockPriceModel")

    # Add tags
    mlflow.set_tag("version", "1.0")
    mlflow.set_tag("type", "LSTM")
    
    # Plot training loss
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig("training_loss_plot.png")  # Save the plot
    mlflow.log_artifact("training_loss_plot.png")  # Log the plot as an artifact
