import sqlite3
import pandas as pd
import joblib
import os

from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

# =======================
# 変更点サマリー
# ① training_data_v2 を使用（翌日陽線ターゲット）
# ② class_weight 不要になる（バランスが改善）
# ③ モデルを stock_model_v2.pkl として保存
# =======================

conn = sqlite3.connect(
    "database/stock.db"
)

df = pd.read_sql(
    """
    SELECT *
    FROM training_data_v2
    """,
    conn
)

conn.close()

# -----------------------
# クラスバランス確認
# -----------------------
print()
print("===== target件数 =====")
counts = df["target"].value_counts()
print(counts)
print(f"陽線率: {counts[1] / len(df) * 100:.1f}%")

print()
print("===== データ件数 =====")
print(df.shape)

# -----------------------
# 特徴量（既存のまま）
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
    subset=features + ["target"]
)

# -----------------------
# 時系列順で分割
# -----------------------
df = df.sort_values("date")

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df  = df.iloc[split_index:]

print()
print(f"学習期間: {train_df['date'].min()} 〜 {train_df['date'].max()}")
print(f"テスト期間: {test_df['date'].min()} 〜 {test_df['date'].max()}")
print(f"学習件数: {len(train_df)}, テスト件数: {len(test_df)}")

X_train = train_df[features]
y_train = train_df["target"]

X_test = test_df[features]
y_test  = test_df["target"]

# -----------------------
# モデル
# ※ バランスが良くなったので class_weight 不要
#   ただし改善しない場合は "balanced" を戻してOK
# -----------------------
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

model.fit(
    X_train,
    y_train
)

# -----------------------
# 評価
# -----------------------
pred = model.predict(X_test)

print()
print("===== classification_report =====")
print(classification_report(y_test, pred))

# -----------------------
# 特徴量重要度
# -----------------------
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print()
print("===== 特徴量重要度 =====")
print(importance.to_string(index=False))

# -----------------------
# 保存
# -----------------------
os.makedirs("data/models", exist_ok=True)

joblib.dump(
    model,
    "data/models/stock_model_v2.pkl"
)

print()
print("モデル保存完了: data/models/stock_model_v2.pkl")
