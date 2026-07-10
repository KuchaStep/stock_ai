import sqlite3
import pandas as pd
import joblib

# -----------------------
# モデル読込
# -----------------------

clf_model = joblib.load(
    "data/models/stock_model_all.pkl"
)

reg_model = joblib.load(
    "data/models/stock_model_regression.pkl"
)

# -----------------------
# DB読込
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

df = df.sort_values(
    "date"
)

# -----------------------
# 分類モデル予測
# -----------------------

df["up_probability"] = (
    clf_model.predict_proba(
        df[features]
    )[:, 1]
)

# -----------------------
# 回帰モデル予測
# -----------------------

df["predicted_return"] = (
    reg_model.predict(
        df[features]
    )
)

# -----------------------
# ハイブリッドスコア
# -----------------------

df["score"] = (
    (df["up_probability"] ** 2) *
    df["predicted_return"].clip(lower=0)
)

# -----------------------
# OOS期間
# -----------------------

split_index = int(
    len(df) * 0.8
)

test_df = df.iloc[
    split_index:
].copy()

# -----------------------
# TOP比較
# -----------------------

top_list = [1, 3, 5, 10]

for top_n in top_list:

    results = []

    for date in sorted(
        test_df["date"].unique()
    ):

        day_df = test_df[
            test_df["date"] == date
        ]

        if len(day_df) < top_n:
            continue

        top_stocks = (
            day_df
            .sort_values(
                "score",
                ascending=False
            )
            .head(top_n)
        )

        if (
            top_stocks["predicted_return"].iloc[0]
            < 0.08
        ):
            continue

        avg_return = (
            top_stocks["future_return"]
            .mean()
        )

        results.append(
            avg_return
        )

    result_df = pd.DataFrame(
        {
            "return": results
        }
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

    print()
    print(f"===== TOP{top_n} =====")

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