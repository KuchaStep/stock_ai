import sqlite3
import pandas as pd
import joblib

# -----------------------
# モデル読み込み
# -----------------------

model = joblib.load(
    "data/models/stock_model_all.pkl"
)

# -----------------------
# DB読み込み
# -----------------------

conn = sqlite3.connect(
    "database/stock.db"
)

df = pd.read_sql(
    """
    SELECT *
    FROM training_data_all
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
# 予測確率
# -----------------------

proba = model.predict_proba(
    df[features]
)

df["up_probability"] = proba[:, 1]

# -----------------------
# 日別バックテスト
# -----------------------

results = []

dates = sorted(
    df["date"].unique()
)

for date in dates:

    day_df = df[
        df["date"] == date
    ]

    if len(day_df) < 3:
        continue

    # 上昇確率TOP3
    top3 = (
        day_df
        .sort_values(
            "up_probability",
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
# 結果DataFrame
# -----------------------

result_df = pd.DataFrame(
    results,
    columns=[
        "date",
        "return"
    ]
)

# -----------------------
# 勝率
# -----------------------

win_rate = (
    result_df["return"] > 0
).mean()

# -----------------------
# 平均リターン
# -----------------------

mean_return = (
    result_df["return"]
    .mean()
)

# -----------------------
# 累積資産
# -----------------------

capital = 100000

for r in result_df["return"]:

    capital *= (
        1 + r
    )

# -----------------------
# 最大ドローダウン
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

print("========== バックテスト結果 ==========")

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
    f"初期資金 : 100000円"
)

print(
    f"最終資産 : {capital:.0f}円"
)

print()

print("===== 直近10件 =====")

print(
    result_df.tail(10)
)