import sqlite3
import pandas as pd
import joblib

from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
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

# -----------------------
# 時系列順
# -----------------------

df = df.sort_values(
    "date"
)

# -----------------------
# 分割
# -----------------------

split_index = int(
    len(df) * 0.8
)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df[features]
y_train = train_df["future_return"]

X_test = test_df[features]
y_test = test_df["future_return"]

# -----------------------
# モデル
# -----------------------

model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42
)

# -----------------------
# 学習
# -----------------------

model.fit(
    X_train,
    y_train
)

# -----------------------
# 予測
# -----------------------

pred = model.predict(
    X_test
)

# -----------------------
# 評価
# -----------------------

mae = mean_absolute_error(
    y_test,
    pred
)

r2 = r2_score(
    y_test,
    pred
)

print()
print("===== 回帰モデル評価 =====")
print()
print(f"MAE : {mae:.4f}")
print(f"R2  : {r2:.4f}")

# -----------------------
# 特徴量重要度
# -----------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print()
print("===== 特徴量重要度 =====")
print(importance)

# -----------------------
# 保存
# -----------------------

joblib.dump(
    model,
    "data/models/stock_model_regression.pkl"
)

print()
print("回帰モデル保存完了")