import sqlite3
import pandas as pd

# features_v2 → training_data_v2b
# ターゲット: 翌日陽線 (future_return > 0)

conn = sqlite3.connect(
    "database/stock.db"
)

features = pd.read_sql(
    """
    SELECT *
    FROM features_v2
    ORDER BY code, date
    """,
    conn
)

all_data = []

for code, df in features.groupby("code"):

    df = df.copy()

    df["future_return"] = (
        df["close"].shift(-1) / df["close"] - 1
    )

    df["target"] = (
        df["future_return"] > 0.0
    ).astype(int)

    df = df.dropna(subset=["future_return"])

    all_data.append(df)

training_data = pd.concat(
    all_data,
    ignore_index=True
)

print()
print("===== ターゲット分布 =====")
counts = training_data["target"].value_counts()
print(counts)
print(f"陽線率: {counts[1] / len(training_data) * 100:.1f}%")

training_data.to_sql(
    "training_data_v2b",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print()
print("training_data_v2b 作成完了")
print(f"総行数: {len(training_data):,}")
