import sqlite3
import pandas as pd
import joblib

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------
# DB読み込み
# -----------------------------

conn = sqlite3.connect(
    "database/stock.db"
)

df = pd.read_sql(
    """
    SELECT *
    FROM training_data
    """,
    conn
)

conn.close()

# -----------------------------
# 使用特徴量
# -----------------------------

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

# -----------------------------
# 欠損除去
# -----------------------------

df = df.dropna(
    subset=features
)

# -----------------------------
# 学習データ
# -----------------------------

X = df[features]
y = df["target"]

# -----------------------------
# 時系列分割
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# -----------------------------
# モデル作成
# -----------------------------

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42
)

# -----------------------------
# 学習
# -----------------------------

model.fit(
    X_train,
    y_train
)

# -----------------------------
# 予測
# -----------------------------

pred = model.predict(
    X_test
)

# -----------------------------
# 評価
# -----------------------------

print(
    classification_report(
        y_test,
        pred
    )
)

# -----------------------------
# 保存
# -----------------------------

joblib.dump(
    model,
    "data/models/stock_model.pkl"
)

print("\nモデル保存完了")

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