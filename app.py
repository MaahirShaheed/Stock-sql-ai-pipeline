import streamlit as st
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Stock SQL AI Pipeline", layout="wide")

st.title("Stock SQL AI Pipeline Dashboard")
st.write("AAPL + SPY feature dashboard with RSI and relative strength")

# Load data
conn = sqlite3.connect("data/market_data.db")
df = pd.read_sql("SELECT * FROM aapl_data", conn)
conn.close()

df["Date"] = pd.to_datetime(df["Date"])
latest = df.iloc[-1]

# Dashboard snapshot
st.subheader("Latest Market Snapshot")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest AAPL Close", f"{latest['Close']:.2f}")
col2.metric("Latest RSI(14)", f"{latest['RSI14']:.2f}")
col3.metric("Latest SPY Return", f"{latest['SPY_Return']:.4f}")
col4.metric("Latest Relative Strength", f"{latest['Relative_Strength']:.4f}")

# Recent data
st.subheader("Recent Data")
st.dataframe(
    df[["Date", "Close", "RSI14", "SPY_Return", "Relative_Strength"]].tail(10),
    use_container_width=True
)

# Charts
st.subheader("AAPL Close Price")
st.line_chart(df.set_index("Date")["Close"])

st.subheader("RSI(14)")
st.line_chart(df.set_index("Date")["RSI14"])

st.subheader("Relative Strength vs SPY")
st.line_chart(df.set_index("Date")["Relative_Strength"])

# =========================
# AI MODEL SECTION
# =========================

feature_cols = [
    "Return",
    "MA10",
    "MA20",
    "Volatility_5",
    "Volume",
    "RSI14",
    "SPY_Return",
    "Relative_Strength"
]

model_df = df.dropna(subset=feature_cols + ["Target"]).copy()

X = model_df[feature_cols]
y = model_df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
log_accuracy = accuracy_score(y_test, y_pred)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Latest prediction
st.header("AI Prediction")

latest_features = model_df[feature_cols].iloc[-1:]
prediction = model.predict(latest_features)[0]
prob = model.predict_proba(latest_features)[0]

if prediction == 1:
    st.success("📈 Model Prediction: STOCK GOING UP")
else:
    st.error("📉 Model Prediction: STOCK GOING DOWN")

col1, col2 = st.columns(2)
col1.metric("Probability Up", f"{prob[1]*100:.1f}%")
col2.metric("Probability Down", f"{prob[0]*100:.1f}%")

# Model performance
st.header("Model Performance")

col3, col4 = st.columns(2)
col3.metric("Logistic Regression Accuracy", f"{log_accuracy*100:.1f}%")
col4.metric("Random Forest Accuracy", f"{rf_accuracy*100:.1f}%")


