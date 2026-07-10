import sqlite3
import pandas as pd
import yfinance as yf

conn = sqlite3.connect(
    "database/stock.db"
)

stocks = pd.read_sql(
    "SELECT * FROM stocks",
    conn
)

all_data = []

for _, row in stocks.iterrows():

    code = row["code"]

    ticker = f"{code}.T"

    print(f"取得中: {ticker}")

    df = yf.download(
        ticker,
        start="2020-01-01",
        progress=False
    )
    if len(df) == 0:
        continue

    df = df.reset_index()

    df.columns = [
        "date",
        "close",
        "high",
        "low",
        "open",
        "volume"
    ]

    df["code"] = code

    df = df[
        [
            "code",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ]

    all_data.append(df)

result = pd.concat(
    all_data,
    ignore_index=True
)

result.to_sql(
    "prices",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print(result.head())
print("保存完了")