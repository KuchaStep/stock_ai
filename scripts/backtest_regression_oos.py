import sqlite3
import pandas as pd
import joblib

# -----------------------
# DB読み込み
# -----------------------

conn = sqlite3.connect(
    "database/stock.db"
)

df = pd.read_sql(
    """
    SELECT *
    FROM training_data_regression
    """,
    conn
)

conn.close()

# -----------------------
# 特徴量
# -----------------------

features = [
    "MA5",
    "MA25",
    "MA75",
    "RSI",
    "return",
    "volume_change",
    "ma5_gap",
    "ma25_gap",
    "ma75_gap",
    "MACD",
    "MACD_signal",
    "volatility"
]

# -----------------------
# 欠損除去
# -----------------------

df = df.dropna(
    subset=features + ["future_return"]
)

# -----------------------
# 時系列順
# -----------------------

df = df.sort_values(
    "date"
)

# -----------------------
# 学習・テスト分割
# -----------------------

split_index = int(
    len(df) * 0.8
)

test_df = df.iloc[
    split_index:
].copy()

# -----------------------
# モデル読み込み
# -----------------------

model = joblib.load(
    "data/models/stock_model_regression.pkl"
)

# -----------------------
# 予測
# -----------------------

test_df["predicted_return"] = (
    model.predict(
        test_df[features]
    )
)

# -----------------------
# 日別バックテスト
# -----------------------

results = []

for date in sorted(
    test_df["date"].unique()
):

    day_df = test_df[
        test_df["date"] == date
    ]

    if len(day_df) < 3:
        continue

    top3 = (
        day_df
        .sort_values(
            "predicted_return",
            ascending=False
        )
        .head(3)
    )

    avg_return = (
        top3["future_return"]
        .mean()
    )

    results.append(
        [
            date,
            avg_return
        ]
    )

# -----------------------
# 集計
# -----------------------

result_df = pd.DataFrame(
    results,
    columns=[
        "date",
        "return"
    ]
)

win_rate = (
    result_df["return"] > 0
).mean()

mean_return = (
    result_df["return"]
    .mean()
)

capital = 100000

for r in result_df["return"]:
    capital *= (
        1 + r
    )

# -----------------------
# DD
# -----------------------

equity = [100000]

current = 100000

for r in result_df["return"]:

    current *= (
        1 + r
    )

    equity.append(current)

equity = pd.Series(equity)

rolling_max = equity.cummax()

drawdown = (
    equity - rolling_max
) / rolling_max

max_drawdown = (
    drawdown.min()
)

# -----------------------
# 出力
# -----------------------

print()
print("===== OOS回帰バックテスト =====")
print()

print(
    f"対象日数 : {len(result_df)}"
)

print(
    f"勝率 : {win_rate*100:.2f}%"
)

print(
    f"平均リターン : {mean_return*100:.2f}%"
)

print(
    f"最大ドローダウン : {max_drawdown*100:.2f}%"
)

print(
    f"最終資産 : {capital:.0f}円"
)