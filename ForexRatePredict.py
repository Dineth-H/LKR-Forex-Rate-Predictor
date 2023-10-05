import tkinter as tk
import pandas as pd
import threading
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(filename='predictor.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data
data = pd.read_csv('USD_LKR Historical Data.csv')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Convert date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Preprocess the Vol. column
data['Vol.'] = data['Vol.'].str.replace('K', '').astype(float) / 1000.0  # Remove 'K' and divide by 1000

# Calculate rolling averages for the USD rate for the past n days
n = 7  # window size
data['USD_Rate_Rolling_Mean'] = data['Price'].rolling(window=n).mean()

# Create lag features for the USD rate
lags = [1, 2, 3]  # number of lags
data[['USD_Rate_Lag_1', 'USD_Rate_Lag_2', 'USD_Rate_Lag_3']] = data['Price'].shift(lags)

# Drop rows with NaN values resulting from rolling mean and lag features
data.dropna(inplace=True)

# Split the data into training and testing sets
X = data[['Open', 'High', 'Low', 'Vol.', 'USD_Rate_Lag_1', 'USD_Rate_Lag_2', 'USD_Rate_Lag_3']]
y = data['Price']  # Assuming 'Price' is the USD exchange rate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model or load a pre-trained model
try:
    # Try to load a pre-trained model
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    # If the model file doesn't exist, create and train a new model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Save the trained model
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics and log them
logging.info(f'Mean Squared Error: {mse}')
logging.info(f'R-squared: {r2}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Define a function for prediction
def predict_usd_rate():
    # Get the date entered by the user from the entry field
    user_date = entry.get()

    try:
        # Convert the user's input date to datetime
        user_date = pd.to_datetime(user_date)

        # Query your dataset to get the features for the user_date
        input_features = data[data['Date'] == user_date][['Open', 'High', 'Low', 'Vol.', 'USD_Rate_Lag_1', 'USD_Rate_Lag_2', 'USD_Rate_Lag_3']]

        # Perform the prediction based on the input features
        predicted_rate = model.predict(input_features)  # Input features should be a DataFrame

        # Update the GUI with the prediction result
        result_label.config(text=f"Predicted USD Rate on {user_date}: ${predicted_rate[0]:.2f}")
    except ValueError:
        result_label.config(text="Invalid date format. Please enter a valid date.")

# Define a function for data visualization
def visualize_data():
    # Create a line plot of historical USD exchange rates
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Price'], label='USD Exchange Rate', color='blue')
    plt.title('Historical USD Exchange Rate')
    plt.xlabel('Date')
    plt.ylabel('USD Exchange Rate')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Create the main application window
app = tk.Tk()
app.title("USD Exchange Rate Predictor")

# Create and place widgets (labels, buttons) in the window
# label
label = tk.Label(app, text="Enter Date:")
label.pack()

# entry field
entry = tk.Entry(app)
entry.pack()

# button for prediction
button = tk.Button(app, text="Predict", command=predict_usd_rate)
button.pack()

# button for data visualization
vis_button = tk.Button(app, text="Visualize Data", command=visualize_data)
vis_button.pack()

# result label
result_label = tk.Label(app, text="")
result_label.pack()

# Start the GUI main loop
app.mainloop()
