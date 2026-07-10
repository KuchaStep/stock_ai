import sqlite3
import pandas as pd
import numpy as np
import joblib
import os
from lightgbm import LGBMRanker

# =======================
# 設定
# =======================

DB_PATH = "database/stock.db"
MODEL_PATH = "data/models/ranker_model.pkl"

# =======================
# データ読み込み
# =======================

conn = sqlite3.connect(DB_PATH)

df = pd.read_sql(
    """
    SELECT *
    FROM training_data_regression
    """,
    conn
)

conn.close()

# =======================
# 特徴量
# =======================

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

# =======================
# 前処理
# =======================

df = df.dropna(subset=features + ["future_return"])
df = df.sort_values(["date", "code"]).reset_index(drop=True)

# =======================
# OOS分割（時系列）
# =======================

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("\n===== データ情報 =====")
print("Train rows:", len(train_df))
print("Test rows :", len(test_df))
print("Train days:", train_df["date"].nunique())
print("Test days :", test_df["date"].nunique())

# =======================
# group作成（最重要）
# =======================

train_group = train_df.groupby("date").size().tolist()
test_group = test_df.groupby("date").size().tolist()

# =======================
# Rankerモデル
# =======================

model = LGBMRanker(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42
)

# =======================
# 学習
# =======================

model.fit(
    train_df[features],
    train_df["future_return"],  # ★これでOK（rank_label不要）
    group=train_group
)

# =======================
# 予測
# =======================

test_df = test_df.copy()
test_df["score"] = model.predict(test_df[features])

# =======================
# TOP評価関数
# =======================

def evaluate_top_n(n):

    results = []

    for date in sorted(test_df["date"].unique()):

        day = test_df[test_df["date"] == date]

        if len(day) < n:
            continue

        top = (
            day.sort_values("score", ascending=False)
            .head(n)
        )

        results.append(top["future_return"].mean())

    result_df = pd.DataFrame({"return": results})

    win_rate = (result_df["return"] > 0).mean()
    mean_return = result_df["return"].mean()

    capital = 100000
    equity = [capital]

    for r in result_df["return"]:
        capital *= (1 + r)
        equity.append(capital)

    equity = pd.Series(equity)
    drawdown = (equity - equity.cummax()) / equity.cummax()

    print(f"\n===== TOP{n} =====")
    print(f"対象日数 : {len(result_df)}")
    print(f"勝率 : {win_rate*100:.2f}%")
    print(f"平均リターン : {mean_return*100:.2f}%")
    print(f"最大ドローダウン : {drawdown.min()*100:.2f}%")
    print(f"最終資産 : {capital:.0f}円")


# =======================
# 評価
# =======================

for n in [1, 3, 5, 10]:
    evaluate_top_n(n)

# =======================
# モデル保存
# =======================

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\nモデル保存完了:", MODEL_PATH)