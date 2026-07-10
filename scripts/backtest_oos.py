import sqlite3
import pandas as pd
import joblib

from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor

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

df = df.dropna(
subset=features + ["future_return"]
)

df = df.sort_values(
"date"
)

# -----------------------

# 分類用ターゲット作成

# -----------------------

df["target"] = (
df["future_return"] > 0.03
).astype(int)

# -----------------------

# OOS分割

# -----------------------

split_index = int(
len(df) * 0.8
)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print()
print("===== データ件数 =====")
print("Train:", len(train_df))
print("Test :", len(test_df))

# -----------------------

# 分類モデル学習

# -----------------------

clf_model = LGBMClassifier(
n_estimators=500,
learning_rate=0.03,
num_leaves=31,
class_weight="balanced",
random_state=42
)

clf_model.fit(
train_df[features],
train_df["target"]
)

# -----------------------

# 回帰モデル学習

# -----------------------

reg_model = LGBMRegressor(
n_estimators=500,
learning_rate=0.03,
num_leaves=31,
random_state=42
)

reg_model.fit(
train_df[features],
train_df["future_return"]
)

# -----------------------

# テスト期間だけ予測

# -----------------------

test_df["up_probability"] = (
clf_model.predict_proba(
test_df[features]
)[:,1]
)

test_df["predicted_return"] = (
reg_model.predict(
test_df[features]
)
)

# -----------------------

# ハイブリッドスコア

# -----------------------

test_df["score"] = (
test_df["up_probability"]
*
test_df["predicted_return"]
)

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

    equity = [100000]

    current = 100000

    for r in result_df["return"]:

        current *= (
            1 + r
        )

        equity.append(current)

    capital = current

    equity = pd.Series(equity)

    rolling_max = equity.cummax()

    drawdown = (
        equity - rolling_max
    ) / rolling_max

    max_dd = drawdown.min()

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
        f"最大ドローダウン : {max_dd*100:.2f}%"
    )
    print(
        f"最終資産 : {capital:.0f}円"
    )
