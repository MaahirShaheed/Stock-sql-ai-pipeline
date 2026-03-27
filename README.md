# Stock SQL AI Pipeline

This project builds an end-to-end financial data pipeline using Python, SQLite, and machine learning to analyze AAPL stock data and predict next-day price movement.

## Features
- Downloads 1 year of AAPL and SPY stock data using yfinance
- Engineers financial features including:
  - daily returns
  - 10-day and 20-day moving averages
  - 5-day volatility
  - RSI(14)
  - SPY benchmark returns
  - relative strength versus SPY
- Stores cleaned and engineered data in a SQLite database
- Trains a logistic regression model to predict whether AAPL will rise the next trading day
- Evaluates performance using accuracy, precision, recall, and F1-score

## Tools Used
- Python
- pandas
- SQLite
- yfinance
- scikit-learn
- matplotlib

## Model Result
The initial logistic regression model achieved about 60% accuracy on test data.

## Project Structure
- `data_pipeline.py` → downloads and engineers AAPL and SPY features
- `model.py` → trains and evaluates the machine learning model
- `data/market_data.db` → SQLite database storing market data and engineered features

## Why This Project Matters
This project demonstrates data engineering, SQL integration, financial feature engineering, benchmark comparison, and machine learning in one workflow.