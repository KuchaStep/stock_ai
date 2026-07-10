import sqlite3
import pandas as pd
import joblib

from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

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

print("\n===== target件数 =====")
print(df["target"].value_counts())

print("\n===== データ件数 =====")
print(df.shape)

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
    subset=features
)

# -----------------------
# 時系列順
# -----------------------

df = df.sort_values(
    ["date"]
)

# -----------------------
# 学習・テスト分割
# -----------------------

split_index = int(
    len(df) * 0.8
)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df[features]
y_train = train_df["target"]

X_test = test_df[features]
y_test = test_df["target"]

# -----------------------
# モデル
# -----------------------

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    class_weight="balanced",
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

print(
    classification_report(
        y_test,
        pred
    )
)

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

print("\n特徴量重要度")
print(importance)

# -----------------------
# 保存
# -----------------------

joblib.dump(
    model,
    "data/models/stock_model_all.pkl"
)

print("\nモデル保存完了")