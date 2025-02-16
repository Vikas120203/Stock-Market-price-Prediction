import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import timedelta

# Load the trained LSTM model
model = load_model("C:\\Users\\Kalisetti Balachandu\\Downloads\\lstm_model.h5")

# Function to download historical stock data
def download_data(ticker="GOOGL"):
    data = yf.download(ticker, start='2004-01-01', end='2024-01-01')
    data.reset_index(inplace=True)
    return data

# Function to preprocess the data: scale and create sequences
def preprocess_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data = data[['Date', 'Close']].set_index('Date')
    scaler = MinMaxScaler(feature_range=(0, 1))
    #data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    scaled_data = scaler.fit_transform(data[['Close']])
    
    # Split the data into training set
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    return train_data, scaler

# Function to create sequences from the training data
def create_sequences(data, seq_length):
    x = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
    return np.array(x)

# Function to predict future stock prices for a specified number of days
def predict_future_price(seq_length, scaler, last_data, n_days=30):
    current_input = last_data.reshape(1, seq_length, 1)
    future_predictions = []

    for _ in range(n_days):
        next_pred = model.predict(current_input)
        future_predictions.append(next_pred[0, 0])
        
        # Update current_input for the next prediction
        next_pred_reshaped = next_pred.reshape(1, 1, 1)  # Reshape to (1, 1, 1)
        current_input = np.append(current_input[:, 1:, :], next_pred_reshaped, axis=1)

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Streamlit app layout and functionality
st.title("📈 Google Stock Price Prediction")
st.sidebar.header("Settings")
st.sidebar.write("This app predicts future Google stock prices using an LSTM model.")

# Load historical data
data = download_data()

# Show historical data
if st.sidebar.checkbox("Show Historical Data"):
    st.subheader("Historical Data")
    st.write(data)

# Plot historical Closing Price data
st.subheader("### Historical Closing Price")
st.line_chart(data['Close'])

# Preprocess data and create sequences for prediction
train_data, scaler = preprocess_data(data)
seq_length = 60
sequences = create_sequences(train_data, seq_length)
last_known_data = sequences[-1]  # Use the last available sequence for predictions

# Set the number of days to predict into the future
n_future_days = st.sidebar.slider("Select number of days to predict into the future", 1, 180)

if st.sidebar.button("Predict"):
    try:
        future_prices = predict_future_price(seq_length, scaler, last_known_data, n_days=n_future_days)

        # Generate dates for the future predictions
        last_date = data['Date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, n_future_days + 1)]

        # Display results
        st.subheader("### Predicted Future Prices")
        prediction_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_prices.flatten()})
        st.write(prediction_df)

        # Plotting the future predictions
        plt.figure(figsize=(10, 5))
        plt.plot(prediction_df['Date'], prediction_df['Predicted Close'], marker='o', linestyle='-', color='orange', label='Predicted Close Price')
        plt.title("Future Predicted Stock Prices")
        plt.xlabel("Date")
        plt.ylabel("Predicted Close Price")
        plt.xticks(rotation=45)
        plt.grid()
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed by Team 4")
st.sidebar.write("Using LSTM Model for Stock Price Prediction")


