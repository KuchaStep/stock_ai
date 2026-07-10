import sqlite3
import pandas as pd

# =======================
# 変更点サマリー
# ① 予測期間: 5日後 → 翌営業日
# ② ターゲット閾値: >3% → >0% (陽線かどうか)
# ③ テーブル名: training_data_all_v2 (既存と共存)
# =======================

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

    # -----------------------
    # ① 翌営業日リターンに変更
    # shift(-1) で翌日終値との比較
    # -----------------------
    future_return = (
        df["close"].shift(-1)
        / df["close"]
        - 1
    )

    df["future_return"] = future_return

    # -----------------------
    # ② 閾値を0%に変更
    # 翌日が陽線なら1、陰線なら0
    # -----------------------
    df["target"] = (
        future_return > 0.0
    ).astype(int)

    df = df.dropna(
        subset=["future_return"]
    )

    all_data.append(df)

training_data = pd.concat(
    all_data,
    ignore_index=True
)

# -----------------------
# クラスバランス確認
# 理想は 0:1 が 45〜55% 程度
# -----------------------
print()
print("===== ターゲット分布 =====")
counts = training_data["target"].value_counts()
print(counts)
print(f"陽線率: {counts[1] / len(training_data) * 100:.1f}%")

# -----------------------
# future_return の分布確認
# -----------------------
print()
print("===== future_return 統計 =====")
print(training_data["future_return"].describe())

training_data.to_sql(
    "training_data_v2",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print()
print("training_data_v2 作成完了")
print("※ 既存の training_data_all はそのまま残しています")
