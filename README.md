# 📈 AI-Powered Stock Prediction Pipeline

This project builds an  data pipeline that analyzes stock market data and predicts next-day price movement using machine learning.

## 🚀 Features

- SQL-based data pipeline storing AAPL + SPY market data
- Feature engineering:
  - RSI (Relative Strength Index)
  - Moving averages (MA10, MA20)
  - Volatility
  - Relative strength vs SPY
- Machine learning models:
  - Logistic Regression
  - Random Forest
- Real-time prediction dashboard built with Streamlit
- Displays:
  - Market snapshot
  - AI prediction (UP / DOWN)
  - Probability confidence
  - Model performance comparison

---

## 🧠 Tech Stack

- Python
- pandas
- scikit-learn
- Streamlit
- SQLite
- matplotlib

---

## 📊 Model Results

- Logistic Regression Accuracy: ~60%
- Random Forest Accuracy: ~47%

Logistic Regression performed better in this financial prediction setting.

---

## 🖥️ Dashboard Preview

![Dashboard](image.png)


---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py