
# 📈 Stock Price Prediction Using Machine Learning (Streamlit App)

## 🚀 Overview

This project is a simple **Stock Price Prediction Web App** built using **Streamlit** and **Machine Learning (Linear Regression)**. It allows users to select any stock symbol (such as **AAPL**, **TCS.NS**, **INFY**) and predict its closing prices based on historical data fetched from **Yahoo Finance**.

The app displays both **actual vs. predicted prices** and provides evaluation metrics like **Mean Squared Error** and **R² Score**.


---

## 💡 Key Features

- 📊 Fetches historical stock prices from **Yahoo Finance**.
- 🔍 Allows user to enter stock ticker, start date, end date.
- 🤖 Predicts next-day closing price using **Linear Regression**.
- 📉 Visualizes **Actual vs. Predicted Prices** on a plot.
- 📝 Displays model evaluation metrics.

---

## 🛠 Technologies Used

- Python
- Streamlit
- yFinance
- Scikit-learn
- Pandas & NumPy
- Matplotlib

---

## 📥 Installation and Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/Stock_Price_Prediction.git
cd Stock_Price_Prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
streamlit run stock_price_app.py
```

---

## 📝 Example Usage:

- Ticker Symbol: `TCS.NS` (for Tata Consultancy Services)
- Start Date: `2018-01-01`
- End Date: `2024-12-31`

➡️ The app will display the stock’s price chart and the Linear Regression predictions.

---

## ⚙ Requirements

- Python 3.x
- Internet connection (to fetch stock data)

---

## 📌 Limitations & Future Improvements

- Currently uses only **Linear Regression** with the previous day's closing price.
- Can be enhanced using:
  - More features (Moving Averages, RSI, MACD, etc.)
  - Advanced models: Random Forest, XGBoost, LSTM
  - Deployment on **Streamlit Cloud** for live access.

---

## 📬 Contact

Created by **Sneha Patil**  
LinkedIn: https://www.linkedin.com/in/sneha-patil-3a96892a5/  

---
