import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

conn = sqlite3.connect("data/market_data.db")
df = pd.read_sql("SELECT * FROM aapl_data", conn)
conn.close()

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

X = df[feature_cols]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
