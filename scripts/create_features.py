import sqlite3
import pandas as pd

conn = sqlite3.connect(
    "database/stock.db"
)

df = pd.read_sql(
    """
    SELECT *
    FROM prices_raw
    ORDER BY Date
    """,
    conn
)

df["MA5"] = df["Close"].rolling(5).mean()
df["MA25"] = df["Close"].rolling(25).mean()
df["MA75"] = df["Close"].rolling(75).mean()

# RSI計算

delta = df["Close"].diff()

gain = delta.where(
    delta > 0,
    0
)

loss = -delta.where(
    delta < 0,
    0
)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["RSI"] = (
    100 - (100 / (1 + rs))
)

df["return"] = df["Close"].pct_change()

df["volume_change"] = (
    df["Volume"].pct_change()
)

df["ma5_gap"] = (
    df["Close"] / df["MA5"]
) - 1

df["ma25_gap"] = (
    df["Close"] / df["MA25"]
) - 1

df["ma75_gap"] = (
    df["Close"] / df["MA75"]
) - 1

print(
    df[
        [
            "Date",
            "Close",
            "MA5",
            "MA25",
            "MA75",
            "RSI",
            "ma75_gap",
            "return"
        ]
    ].tail()
)

df.to_sql(
    "features",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("features作成完了")