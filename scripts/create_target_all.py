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

    future_return = (
        df["close"].shift(-5)
        / df["close"]
        - 1
    )

    df["future_return"] = future_return

    df["target"] = (
        future_return > 0.03
    ).astype(int)

    df = df.dropna(
        subset=["future_return"]
    )

    all_data.append(df)

training_data = pd.concat(
    all_data,
    ignore_index=True
)

print(
    training_data["target"]
    .value_counts()
)

training_data.to_sql(
    "training_data_all",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print(
    "\ntraining_data_all作成完了"
)