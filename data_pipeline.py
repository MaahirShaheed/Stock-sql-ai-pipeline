import pandas as pd
import yfinance as yf
import sqlite3

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Download AAPL and SPY data
aapl = yf.download("AAPL", period="1y", interval="1d")
spy = yf.download("SPY", period="1y", interval="1d")

# Flatten columns if needed
if isinstance(aapl.columns, pd.MultiIndex):
    aapl.columns = aapl.columns.get_level_values(0)

if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

# Reset index
aapl = aapl.reset_index()
spy = spy.reset_index()

# Keep only needed SPY columns
spy = spy[["Date", "Close"]].rename(columns={"Close": "SPY_Close"})

# Merge AAPL with SPY on Date
df = pd.merge(aapl, spy, on="Date", how="inner")

# AAPL features
df["Return"] = df["Close"].pct_change()
df["MA10"] = df["Close"].rolling(10).mean()
df["MA20"] = df["Close"].rolling(20).mean()
df["Volatility_5"] = df["Return"].rolling(5).std()
df["RSI14"] = compute_rsi(df["Close"], window=14)

# SPY comparison features
df["SPY_Return"] = df["SPY_Close"].pct_change()
df["Relative_Strength"] = df["Return"] - df["SPY_Return"]

# Target: 1 if next day close is higher, else 0
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# Drop missing rows
df = df.dropna()

# Save to SQLite
conn = sqlite3.connect("data/market_data.db")
df.to_sql("aapl_data", conn, if_exists="replace", index=False)
conn.close()

print("Updated data with RSI and SPY comparison saved successfully.")
print(df.head())