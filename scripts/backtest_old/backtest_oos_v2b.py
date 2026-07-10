import sqlite3
import pandas as pd
from lightgbm import LGBMClassifier

# =======================
# 旧特徴量12個 + 新特徴量9個 = 計21個で評価
# =======================

FEATURES_OLD = [
    "MA5", "MA25", "MA75",
    "RSI", "return", "volume_change",
    "ma5_gap", "ma25_gap", "ma75_gap",
    "MACD", "MACD_signal", "volatility"
]

FEATURES_NEW = [
    "high52w_gap",      # 52週高値からの距離
    "low52w_gap",       # 52週安値からの距離
    "volume_zscore",    # 出来高Zスコア
    "bb_pct",           # ボリンジャー %B
    "atr_pct",          # ATR(株価比)
    "mom3",             # 3日モメンタム
    "mom10",            # 10日モメンタム
    "mom20",            # 20日モメンタム
    "return_vs_market", # 市場平均との騰落差
]

ALL_FEATURES = FEATURES_OLD + FEATURES_NEW

conn = sqlite3.connect(
    "database/stock.db"
)

df = pd.read_sql(
    "SELECT * FROM training_data_v2b",
    conn
)

conn.close()

df = df.dropna(subset=ALL_FEATURES + ["future_return", "target"])
df = df.sort_values("date").reset_index(drop=True)

# OOS分割
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index].copy()
test_df  = df.iloc[split_index:].copy()

print()
print(f"学習期間 : {train_df['date'].min()} 〜 {train_df['date'].max()}")
print(f"テスト期間: {test_df['date'].min()} 〜 {test_df['date'].max()}")
print()

# モデル学習
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

model.fit(train_df[ALL_FEATURES], train_df["target"])

test_df["up_probability"] = (
    model.predict_proba(test_df[ALL_FEATURES])[:, 1]
)

# -----------------------
# up_probability 分布（改善確認）
# std が 0.057 より大きくなれば特徴量が効いている
# -----------------------
print("===== up_probability 統計 =====")
prob_stats = test_df["up_probability"].describe()
print(prob_stats.round(4))
print()

old_std = 0.0567  # 前回の値
new_std = prob_stats["std"]
diff    = new_std - old_std
sign    = "↑改善" if diff > 0 else "→変化なし"
print(f"前回 std: {old_std:.4f}  今回 std: {new_std:.4f}  {sign}")
print()

# バックテスト
bench_avg = (
    test_df.groupby("date")["future_return"]
    .mean().mean()
)

for top_n in [1, 3, 5, 10]:

    results = []

    for date in sorted(test_df["date"].unique()):
        day = test_df[test_df["date"] == date]
        if len(day) < top_n:
            continue
        top = day.sort_values("up_probability", ascending=False).head(top_n)
        results.append(top["future_return"].mean())

    rdf      = pd.DataFrame({"return": results})
    win_rate = (rdf["return"] > 0).mean()
    mean_r   = rdf["return"].mean()

    capital = 100000
    equity  = [capital]
    for r in rdf["return"]:
        capital *= (1 + r)
        equity.append(capital)

    equity   = pd.Series(equity)
    max_dd   = ((equity - equity.cummax()) / equity.cummax()).min()

    print(f"===== TOP{top_n} =====")
    print(f"勝率             : {win_rate*100:.2f}%")
    print(f"平均リターン     : {mean_r*100:.3f}%  (ベンチマーク: {bench_avg*100:.3f}%)")
    print(f"超過リターン     : {(mean_r - bench_avg)*100:.3f}%")
    print(f"最大ドローダウン : {max_dd*100:.2f}%")
    print(f"最終資産         : {capital:.0f}円")
    print()

# 特徴量重要度（上位10個）
importance = pd.DataFrame({
    "feature":   ALL_FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("===== 特徴量重要度 TOP10 =====")
print(importance.head(10).to_string(index=False))
