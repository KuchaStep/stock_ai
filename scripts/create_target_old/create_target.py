import sqlite3
import pandas as pd

# DB接続
conn = sqlite3.connect(
    "database/stock.db"
)

# featuresテーブル読み込み
df = pd.read_sql(
    """
    SELECT *
    FROM features
    ORDER BY Date
    """,
    conn
)

# -----------------------------
# 5営業日後のリターンを計算
# -----------------------------
future_return = (
    df["Close"].shift(-5)
    / df["Close"]
    - 1
)

# -----------------------------
# target作成
# 5日後に3%以上上昇なら1
# それ以外0
# -----------------------------
df["target"] = (
    future_return > 0.03
).astype(int)

df["future_return"] = future_return

df = df.dropna(
    subset=["future_return"]
)

# future_returnも保存しておく
df["future_return"] = future_return

# -----------------------------
# 確認
# -----------------------------
print("\n=== target確認 ===")
print(
    df[
        [
            "Date",
            "Close",
            "future_return",
            "target"
        ]
    ].tail(15)
)

print("\n=== target件数 ===")
print(
    df["target"]
    .value_counts()
)

# -----------------------------
# 学習用テーブル保存
# -----------------------------
df.to_sql(
    "training_data",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("\ntraining_data作成完了")