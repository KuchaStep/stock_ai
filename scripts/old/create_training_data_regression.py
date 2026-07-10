import sqlite3
import pandas as pd

conn = sqlite3.connect(
    "database/stock.db"
)

features = pd.read_sql(
    """
    SELECT *
    FROM features_all
    ORDER BY code, date
    """,
    conn
)

all_data = []

for code, df in features.groupby("code"):

    print(f"処理中: {code}")

    df = df.copy()

    # 5営業日後
    df["future_return"] = (
        df["close"].shift(-5)
        / df["close"]
        - 1
    )

    df = df.dropna(
        subset=["future_return"]
    )

    all_data.append(df)

training_data = pd.concat(
    all_data,
    ignore_index=True
)

training_data.to_sql(
    "training_data_regression",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print()
print(training_data["future_return"].describe())
print()
print("training_data_regression作成完了")