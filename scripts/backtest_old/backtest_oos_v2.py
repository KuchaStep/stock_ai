import sqlite3
import pandas as pd
import joblib

from lightgbm import LGBMClassifier

# =======================
# 変更点サマリー
# ① training_data_v2 を使用（翌日陽線）
# ② train/test を時系列80/20で分割してOOS評価
# ③ 分類モデルのみ（翌日予測は回帰より分類が安定）
# ④ スコア = up_probability で TOP比較
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
    subset=features + ["future_return", "target"]
)

df = df.sort_values("date").reset_index(drop=True)

# -----------------------
# OOS分割（時系列）
# -----------------------
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df  = df.iloc[split_index:].copy()

print()
print("===== データ情報 =====")
print(f"学習期間 : {train_df['date'].min()} 〜 {train_df['date'].max()}")
print(f"テスト期間: {test_df['date'].min()} 〜 {test_df['date'].max()}")
print(f"学習件数  : {len(train_df)}")
print(f"テスト件数: {len(test_df)}")
print()

# -----------------------
# OOS内のクラスバランス確認
# -----------------------
print("===== テスト期間のターゲット分布 =====")
counts = test_df["target"].value_counts()
print(counts)
print(f"陽線率: {counts[1] / len(test_df) * 100:.1f}%")
print()

# -----------------------
# 分類モデル学習
# -----------------------
clf_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

clf_model.fit(
    train_df[features],
    train_df["target"]
)

# -----------------------
# テスト期間を予測
# -----------------------
test_df["up_probability"] = (
    clf_model.predict_proba(test_df[features])[:, 1]
)

# -----------------------
# TOP比較バックテスト
# ※ future_return は「翌日の実際のリターン」
# -----------------------
top_list = [1, 3, 5, 10]

for top_n in top_list:

    results = []

    for date in sorted(test_df["date"].unique()):

        day_df = test_df[test_df["date"] == date]

        if len(day_df) < top_n:
            continue

        top_stocks = (
            day_df
            .sort_values("up_probability", ascending=False)
            .head(top_n)
        )

        avg_return = top_stocks["future_return"].mean()

        results.append(avg_return)

    result_df = pd.DataFrame({"return": results})

    win_rate    = (result_df["return"] > 0).mean()
    mean_return = result_df["return"].mean()

    capital = 100000
    equity  = [capital]

    for r in result_df["return"]:
        capital *= (1 + r)
        equity.append(capital)

    equity   = pd.Series(equity)
    drawdown = (equity - equity.cummax()) / equity.cummax()
    max_dd   = drawdown.min()

    # ベンチマーク（全銘柄平均）
    bench_avg = (
        test_df.groupby("date")["future_return"]
        .mean()
        .mean()
    )

    print(f"===== TOP{top_n} =====")
    print(f"対象日数         : {len(result_df)}")
    print(f"勝率             : {win_rate*100:.2f}%")
    print(f"平均リターン     : {mean_return*100:.3f}%")
    print(f"最大ドローダウン : {max_dd*100:.2f}%")
    print(f"最終資産         : {capital:.0f}円")
    print(f"ベンチマーク平均 : {bench_avg*100:.3f}%  ← これより高ければモデルが効いている")
    print()

# -----------------------
# 上昇確率分布（確認用）
# -----------------------
print("===== up_probability 統計 =====")
print(test_df["up_probability"].describe().round(4))
print()
print("※ 平均が0.5付近、分散が大きいほど分類が機能している")
