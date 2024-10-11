import pandas as pd

# Load the dataset
data = pd.read_csv('../Data/dataset_RL.csv')

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Create a new column for Year and DayOfYear
data['Year'] = data['Date'].dt.year
data['DayOfYear'] = data['Date'].dt.dayofyear

# Calculate daily percentage changes for Open and Close prices
data['Open_Pct_Change'] = data['Open'].pct_change()
data['Close_Pct_Change'] = data['Close'].pct_change()

# Group by DayOfYear and calculate average percentage changes
average_pct_change_open = data.groupby('DayOfYear')['Open_Pct_Change'].mean().reset_index()
average_pct_change_open.columns = ['DayOfYear', 'Avg_Open_Pct_Change']

average_pct_change_close = data.groupby('DayOfYear')['Close_Pct_Change'].mean().reset_index()
average_pct_change_close.columns = ['DayOfYear', 'Avg_Close_Pct_Change']

# Merge the average percentage changes
average_pct_change = pd.merge(average_pct_change_open, average_pct_change_close, on='DayOfYear')

# Prepare for prediction for the year 2023
predicted_open_prices = []
predicted_close_prices = []

# Loop through each day of the year 2023
for index in range(1, 366):  # Considering leap year
    avg_open_pct = average_pct_change.loc[average_pct_change['DayOfYear'] == index, 'Avg_Open_Pct_Change'].values
    avg_close_pct = average_pct_change.loc[average_pct_change['DayOfYear'] == index, 'Avg_Close_Pct_Change'].values

    if avg_open_pct.size > 0 and avg_close_pct.size > 0:
        avg_open_pct = avg_open_pct[0]
        avg_close_pct = avg_close_pct[0]

        # Predict Open and Close prices based on previous day's prices
        if index == 1:
            predicted_open = data[data['Year'] == 2022]['Open'].iloc[-1] * (1 + avg_open_pct)
            predicted_close = data[data['Year'] == 2022]['Close'].iloc[-1] * (1 + avg_close_pct)
        else:
            predicted_open = predicted_open_prices[-1] * (1 + avg_open_pct)
            predicted_close = predicted_close_prices[-1] * (1 + avg_close_pct)
        
        predicted_open_prices.append(predicted_open)
        predicted_close_prices.append(predicted_close)
    else:
        # If there is no data for that day, use the last known price
        predicted_open_prices.append(predicted_open_prices[-1] if predicted_open_prices else None)
        predicted_close_prices.append(predicted_close_prices[-1] if predicted_close_prices else None)

data = pd.read_csv('../Data/dataset_RL.csv')

# Ensure the 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')

# Create a new column for Year and DayOfYear
data['Year'] = data['Date'].dt.year
data['DayOfYear'] = data['Date'].dt.dayofyear

# Ensure the specific date is in the correct format
specific_date = pd.to_datetime('2023-05-20')  # Change to your desired date




print(f"Predicted Open Price on {specific_date.date()}: {predicted_open_prices[0] if predicted_open_prices else 'N/A'}")
print(f"Predicted Close Price on {specific_date.date()}: {predicted_close_prices[0] if predicted_close_prices else 'N/A'}")
