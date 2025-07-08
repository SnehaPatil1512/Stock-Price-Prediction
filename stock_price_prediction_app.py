import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit Title and Description
st.title('📈 Stock Price Prediction Using Linear Regression')
st.write("""
This app predicts the **next day's closing price** of a stock using **Linear Regression** based on historical data.
""")

# Sidebar: User Inputs
st.sidebar.header('User Input')
ticker = st.sidebar.text_input('Enter Stock Ticker (e.g., AAPL, TCS.NS)', value='TCS.NS')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2018-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2024-12-31'))

# Fetch Data
@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, auto_adjust=True)

data = load_data(ticker, start_date, end_date)

# Show Raw Data
st.subheader(f'{ticker} Stock Closing Price')
st.line_chart(data['Close'])

# Check Data Length
if len(data) < 2:
    st.warning("Not enough data to build the model. Try selecting a wider date range.")
    st.stop()

# Prepare Data
data['Previous_Close'] = data['Close'].shift(1)
data.dropna(inplace=True)

X = data[['Previous_Close']]
y = data['Close']

# Split Data (80-20, no shuffle because it's time series)
split_idx = int(len(data) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f'**Mean Squared Error:** {mse:.2f}')
st.write(f'**R-squared Score:** {r2:.2f}')

# Plot Actual vs Predicted
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(y_test.values, label='Actual Price', color='blue')
ax.plot(y_pred, label='Predicted Price', color='red')
ax.set_title(f'{ticker} Actual vs Predicted Closing Prices')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend()

st.pyplot(fig)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit Title and Description
st.title('📈 Stock Price Prediction Using Linear Regression')
st.write("""
This app predicts the **next day's closing price** of a stock using **Linear Regression** based on historical data.
""")

# Sidebar: User Inputs
st.sidebar.header('User Input')
ticker = st.sidebar.text_input('Enter Stock Ticker (e.g., AAPL, TCS.NS)', value='TCS.NS')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2018-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2024-12-31'))

# Fetch Data
@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, auto_adjust=True)

data = load_data(ticker, start_date, end_date)

# Show Raw Data
st.subheader(f'{ticker} Stock Closing Price')
st.line_chart(data['Close'])

# Check Data Length
if len(data) < 2:
    st.warning("Not enough data to build the model. Try selecting a wider date range.")
    st.stop()

# Prepare Data
data['Previous_Close'] = data['Close'].shift(1)
data.dropna(inplace=True)

X = data[['Previous_Close']]
y = data['Close']

# Split Data (80-20, no shuffle because it's time series)
split_idx = int(len(data) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f'**Mean Squared Error:** {mse:.2f}')
st.write(f'**R-squared Score:** {r2:.2f}')

# Plot Actual vs Predicted
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(y_test.values, label='Actual Price', color='blue')
ax.plot(y_pred, label='Predicted Price', color='red')
ax.set_title(f'{ticker} Actual vs Predicted Closing Prices')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend()

st.pyplot(fig)
