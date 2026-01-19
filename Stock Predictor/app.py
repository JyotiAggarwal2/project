# Importing necessary modules
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px

# Function to load data from Yahoo Finance
def load_data(ticker):
    try:
        data = yf.download(ticker, start="2010-01-01", end="2023-12-31")
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}. Please try again.")
        return None

# Feature engineering: Adding moving averages
def add_features(data):
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data = data.dropna()  # Drop NaN values created by rolling windows
    return data

# Train a Linear Regression model
def train_model(data):
    X = data[['MA20', 'MA50']].values
    y = data['Close'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, rmse

# Predict the next day's price
def predict_future(model, last_row):
    future_price = model.predict(last_row[['MA20', 'MA50']].values)
    return float(future_price[0])  # Ensure scalar value

# Plot the closing price using Plotly
def plot_closing_price(data, ticker):
    # Ensure proper data types
    data['Date'] = pd.to_datetime(data['Date'])
    data['Close'] = data['Close'].astype(float)
    
    # Plot
    fig = px.line(data, x='Date', y='Close', title=f'{ticker} Closing Price')
    st.plotly_chart(fig)

# Streamlit app interface

st.title('🤑 EqiPredict - Stock Predictor 💰')
st.subheader('Highlighting its focus on equity and prediction!')

st.markdown(
    """
    <style>
    .footer {
        font-size: 14px;
        color: gray;
        margin-top: 30px;
        text-align: center;
    }
    """,
    unsafe_allow_html=True
)

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, GOOG)", "AAPL")

if ticker:
    # Load data
    data = load_data(ticker)
    if data is not None and not data.empty:
        st.write("Recent Data", data.tail())
        
        # Feature engineering
        data = add_features(data)
        
        # Check if there are enough data points after feature engineering
        if len(data) < 10:
            st.warning("Not enough data points after feature engineering. Please choose a different stock.")
        else:
            st.write("Data with Features", data.tail())
            
            # Train model
            model, rmse = train_model(data)
            st.write(f"Model RMSE: {rmse:.2f}")
            
            # Predict next day price
            future_price = predict_future(model, data.iloc[[-1]])
            st.write(f"Predicted next day price for {ticker}: ${future_price:.2f}")
            
    else:
        st.error("No data found for the given ticker. Please try again.")

# Add custom footer
st.markdown(
    """
    <div class="footer">
        © 2024 Stock Predictor | Developed by Mohit Aggarwal
    </div>
    """,
    unsafe_allow_html=True
)

