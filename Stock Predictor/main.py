# Importing necessary modules
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import yfinance as yf  # To fetch stock market data from Yahoo Finance
import streamlit as st  # For building interactive web applications
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.linear_model import LinearRegression  # For linear regression modeling
from sklearn.metrics import mean_squared_error  # To calculate the Root Mean Squared Error (RMSE)
import plotly.express as px  # For creating interactive visualizations

# Function to load data from Yahoo Finance
def load_data(ticker):
    """
    Downloads historical stock data for the given ticker symbol.

    Parameters:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').

    Returns:
        DataFrame: Stock data with columns like Date, Open, High, Low, Close, and Volume.
    """
    try:
        # Fetch data for the specified date range
        data = yf.download(ticker, start="2010-01-01", end="2023-12-31")
        data.reset_index(inplace=True)  # Reset the index to make 'Date' a column
        return data
    except Exception as e:
        # Display error in the Streamlit app if data loading fails
        st.error(f"Error loading data for {ticker}. Please try again.")
        return None

# Feature engineering: Adding moving averages
def add_features(data):
    """
    Adds 20-day and 50-day moving averages to the data.

    Parameters:
        data (DataFrame): The stock data.

    Returns:
        DataFrame: Updated data with added moving average columns.
    """
    data['MA20'] = data['Close'].rolling(window=20).mean()  # 20-day moving average
    data['MA50'] = data['Close'].rolling(window=50).mean()  # 50-day moving average
    data = data.dropna()  # Remove rows with NaN values created by rolling averages
    return data

# Train a Linear Regression model
def train_model(data):
    """
    Trains a linear regression model to predict stock closing prices.

    Parameters:
        data (DataFrame): The stock data with features.

    Returns:
        tuple: The trained model and RMSE of predictions on the test set.
    """
    # Features (X) and target variable (y)
    X = data[['MA20', 'MA50']].values
    y = data['Close'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set and calculate RMSE
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, rmse

# Predict the next day's price
def predict_future(model, last_row):
    """
    Predicts the stock closing price for the next day.

    Parameters:
        model (LinearRegression): The trained regression model.
        last_row (DataFrame): The last row of the dataset with features.

    Returns:
        float: Predicted closing price.
    """
    future_price = model.predict(last_row[['MA20', 'MA50']].values)  # Predict using features
    return float(future_price[0])  # Return as a scalar value

# Plot the closing price using Plotly
def plot_closing_price(data, ticker):
    """
    Plots the stock's historical closing prices.

    Parameters:
        data (DataFrame): The stock data with 'Date' and 'Close' columns.
        ticker (str): The stock ticker symbol.

    Displays:
        Interactive Plotly line chart in Streamlit.
    """
    # Ensure proper data types for plotting
    data['Date'] = pd.to_datetime(data['Date'])
    data['Close'] = data['Close'].astype(float)
    
    # Create the plot
    fig = px.line(data, x='Date', y='Close', title=f'{ticker} Closing Price')
    st.plotly_chart(fig)

# Streamlit app interface
st.title('🤑 EqiPredict - Stock Predictor 💰')
st.subheader('Highlighting its focus on equity and prediction!')

# Add custom CSS styling
st.markdown(
    """
    <style>
    .footer {
        font-size: 14px;
        color: gray;
        margin-top: 30px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Input field for the stock ticker symbol
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, GOOG)", "AAPL")

if ticker:
    # Load data for the given ticker
    data = load_data(ticker)
    if data is not None and not data.empty:
        st.write("Recent Data", data.tail())  # Display the last few rows of data

        # Add moving average features
        data = add_features(data)

        # Ensure there are enough data points after feature engineering
        if len(data) < 10:
            st.warning("Not enough data points after feature engineering. Please choose a different stock.")
        else:
            st.write("Data with Features", data.tail())  # Display processed data

            # Train the model and calculate RMSE
            model, rmse = train_model(data)
            st.write(f"Model RMSE: {rmse:.2f}")  # Display model performance

            # Predict the next day's closing price
            future_price = predict_future(model, data.iloc[[-1]])
            st.write(f"Predicted next day price for {ticker}: ${future_price:.2f}")
    else:
        st.error("No data found for the given ticker. Please try again.")

# Add custom footer to the Streamlit app
st.markdown(
    """
    <div class="footer">
        © 2024 Stock Predictor | Developed by Mohit Aggarwal
    </div>
    """,
    unsafe_allow_html=True
)
