import yfinance as yf
import sqlite3

ticker = "7203.T"

df = yf.download(
    ticker,
    start="2020-01-01"
)

print(df.head())

conn = sqlite3.connect(
    "database/stock.db"
)

#DBに入れやすい形へ変換
df = df.reset_index()

df.columns = [
    "Date",
    "Close",
    "High",
    "Low",
    "Open",
    "Volume"
]

print(df.columns)

#DB保存
df.to_sql(
    "prices_raw",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("保存完了")