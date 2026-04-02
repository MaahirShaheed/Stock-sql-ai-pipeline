import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

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

df = df.dropna(subset=feature_cols + ["Target"])

X = df[feature_cols]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

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

df = df.dropna(subset=feature_cols + ["Target"])

X = df[feature_cols]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)



print("\nRandom Forest Accuracy:", rf_accuracy)
print(classification_report(y_test, rf_pred))