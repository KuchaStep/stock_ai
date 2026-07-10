import sqlite3
import pandas as pd
from lightgbm import LGBMClassifier

# =======================
# 特徴量リスト（計31個）
# =======================

FEATURES_BASE = [
    "MA5", "MA25", "MA75",
    "RSI", "return", "volume_change",
    "ma5_gap", "ma25_gap", "ma75_gap",
    "MACD", "MACD_signal", "volatility"
]

FEATURES_V2 = [
    "high52w_gap", "low52w_gap",
    "volume_zscore", "bb_pct", "atr_pct",
    "mom3", "mom10", "mom20",
    "return_vs_market"
]

FEATURES_V3_VOLUME = [
    "vpt_slope",        # VPT傾き（資金流入トレンド）
    "volume_up_ratio",  # 上昇日の出来高比率
    "volume_spike",     # 出来高スパイク
    "obv_slope",        # OBV傾き
]

FEATURES_V3_PATTERN = [
    "gap_up",           # 窓開けギャップ
    "upper_shadow",     # 上ヒゲ比率
    "lower_shadow",     # 下ヒゲ比率
    "body_ratio",       # 実体比率
    "consecutive_up",   # 連続上昇日数
    "consecutive_down", # 連続下落日数
]

ALL_FEATURES = (
    FEATURES_BASE +
    FEATURES_V2 +
    FEATURES_V3_VOLUME +
    FEATURES_V3_PATTERN
)

print(f"使用特徴量数: {len(ALL_FEATURES)}個")

# -----------------------
# データ読み込み
# -----------------------
conn = sqlite3.connect("database/stock.db")

df = pd.read_sql(
    "SELECT * FROM training_data_v3",
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
print(f"学習件数  : {len(train_df):,}  テスト件数: {len(test_df):,}")
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

# up_probability 分布確認
print("===== up_probability 統計 =====")
prob_stats = test_df["up_probability"].describe()
print(prob_stats.round(4))

prev_std = 0.0592  # v2の値
new_std  = prob_stats["std"]
print(f"\n前回(v2) std: {prev_std:.4f}  今回(v3) std: {new_std:.4f}  {'↑改善' if new_std > prev_std else '→変化なし'}")
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

    equity = pd.Series(equity)
    max_dd = ((equity - equity.cummax()) / equity.cummax()).min()

    print(f"===== TOP{top_n} =====")
    print(f"勝率             : {win_rate*100:.2f}%")
    print(f"平均リターン     : {mean_r*100:.3f}%  (ベンチマーク: {bench_avg*100:.3f}%)")
    print(f"超過リターン     : {(mean_r - bench_avg)*100:.3f}%")
    print(f"最大ドローダウン : {max_dd*100:.2f}%")
    print(f"最終資産         : {capital:.0f}円")
    print()

# 特徴量重要度
importance = pd.DataFrame({
    "feature":    ALL_FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("===== 特徴量重要度 TOP15 =====")
print(importance.head(15).to_string(index=False))
print()

# v3新特徴量だけの順位確認
v3_all = FEATURES_V3_VOLUME + FEATURES_V3_PATTERN
v3_imp = importance[importance["feature"].isin(v3_all)]
print("===== v3新特徴量の重要度 =====")
print(v3_imp.to_string(index=False))
