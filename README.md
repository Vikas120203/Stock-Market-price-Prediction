📈 Google Stock Price Prediction using LSTM




🚀 Predict future stock prices of Google (GOOGL) using LSTM (Long Short-Term Memory) neural networks. This project leverages historical stock data, time series forecasting, and deep learning techniques to achieve high accuracy in financial predictions.

Financial markets are highly volatile, and predicting stock prices is a challenging task. This project builds a deep learning-based stock price prediction model using LSTM, a powerful recurrent neural network designed for sequential data.

📊 Stock Ticker Analyzed: Google (GOOGL)
🕒 Time Series Forecasting: Uses past stock prices to predict future trends
📌 Technologies Used: TensorFlow, Keras, Pandas, NumPy, Matplotlib
📂 Dataset Source: Yahoo Finance

✨ Features
✅ Uses 20 years of stock data (2004-2024) for training
✅ LSTM-based deep learning model for accurate predictions
✅ Feature Engineering: Extracts trends, seasonality, and volatility
✅ Data Normalization with MinMaxScaler for better performance
✅ Visualizations of Stock Trends, Moving Averages & Predictions
✅ Compares LSTM performance with ARIMA, SARIMA, and ML models

Data Preprocessing Techniques:

Convert date to numerical features (Year, Month, Day, Day of Week, IsMonthStart/End)
Normalize stock prices using MinMaxScaler
Sliding window technique to structure time-series input for LSTM

 Methodology
1️⃣ Data Collection & Preprocessing
✅ Yahoo Finance API for fetching historical stock prices
✅ Feature extraction – date-based and stock-related indicators
✅ Data normalization using MinMaxScaler

2️⃣ Model Building - LSTM
✅ LSTM layers to learn long-term dependencies in stock prices
✅ Dropout layers to prevent overfitting
✅ Adam optimizer for faster convergence
✅ Mean Squared Error (MSE) loss function

3️⃣ Evaluation & Comparison
✅ Evaluated LSTM against ARIMA, SARIMA, and ML models (XGBoost, Random Forest, SVR)
✅ Metrics used: MSE, RMSE, Accuracy (%)


Model Performance
Model	MSE	RMSE	Accuracy (%)
LSTM	0.00021	0.0145	98.0%
ARIMA	0.00032	0.0179	87.3%
SARIMA	0.00029	0.0169	88.1%
XGBoost	0.00035	0.0187	85.6%
Random Forest	0.00039	0.0198	83.4%
SVR	0.00041	0.0202	82.1%
