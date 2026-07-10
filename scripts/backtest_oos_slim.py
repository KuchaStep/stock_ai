import sqlite3
import pandas as pd
from lightgbm import LGBMClassifier

# =======================
# v3から削除する特徴量（重要度が低くノイズになっている）
# consecutive_up   : 72
# consecutive_down : 55
# MA5/MA25/MA75    : v2時点で圏外
# MACD/MACD_signal : v2時点で圏外
# bb_pct           : v2時点で圏外
# mom20            : 低め
# =======================

FEATURES_SLIM = [
    # 価格モメンタム系（効いているもののみ）
    "return",
    "return_vs_market",
    "mom3",
    "mom10",

    # 出来高系（全部効いている）
    "volume_change",
    "volume_zscore",
    "volume_up_ratio",
    "volume_spike",
    "obv_slope",
    "vpt_slope",

    # 価格位置系
    "high52w_gap",
    "low52w_gap",
    "atr_pct",
    "volatility",

    # ローソク足パターン系
    "gap_up",
    "upper_shadow",
    "lower_shadow",
    "body_ratio",

    # RSI（単体では弱いが他と組み合わせで効く可能性）
    "RSI",
]

print(f"使用特徴量数: {len(FEATURES_SLIM)}個  (v3の31個 → {len(FEATURES_SLIM)}個に削減)")

# -----------------------
# データ読み込み
# -----------------------
conn = sqlite3.connect("database/stock.db")

df = pd.read_sql(
    "SELECT * FROM training_data_v3",
    conn
)

conn.close()

df = df.dropna(subset=FEATURES_SLIM + ["future_return", "target"])
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

model.fit(train_df[FEATURES_SLIM], train_df["target"])

test_df["up_probability"] = (
    model.predict_proba(test_df[FEATURES_SLIM])[:, 1]
)

# up_probability 分布確認
print("===== up_probability 統計 =====")
prob_stats = test_df["up_probability"].describe()
print(prob_stats.round(4))

prev_std = 0.0585  # v3の値
new_std  = prob_stats["std"]
print(f"\n前回(v3) std: {prev_std:.4f}  今回(slim) std: {new_std:.4f}  {'↑改善' if new_std > prev_std else '↓低下' if new_std < prev_std - 0.001 else '→ほぼ同じ'}")
print()

# バックテスト
bench_avg = (
    test_df.groupby("date")["future_return"]
    .mean().mean()
)

print(f"ベンチマーク（全銘柄平均）: {bench_avg*100:.3f}%")
print()

best_capital = 0
best_top_n   = 0

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

    if capital > best_capital:
        best_capital = capital
        best_top_n   = top_n

    # v3との比較
    v3_results = {
        1:  {"capital": 167234, "excess": 0.102},
        3:  {"capital": 143208, "excess": 0.025},
        5:  {"capital": 153233, "excess": 0.047},
        10: {"capital": 137233, "excess": 0.002},
    }
    v3_cap    = v3_results[top_n]["capital"]
    cap_diff  = capital - v3_cap
    cap_sign  = "↑" if cap_diff > 0 else "↓"

    print(f"===== TOP{top_n} =====")
    print(f"勝率             : {win_rate*100:.2f}%")
    print(f"平均リターン     : {mean_r*100:.3f}%  (ベンチマーク: {bench_avg*100:.3f}%)")
    print(f"超過リターン     : {(mean_r - bench_avg)*100:.3f}%")
    print(f"最大ドローダウン : {max_dd*100:.2f}%")
    print(f"最終資産         : {capital:.0f}円  (v3比: {cap_sign}{abs(cap_diff):.0f}円)")
    print()

print(f"★ 最良構成: TOP{best_top_n}  最終資産: {best_capital:.0f}円")
print()

# 特徴量重要度
importance = pd.DataFrame({
    "feature":    FEATURES_SLIM,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("===== 特徴量重要度（スリム版） =====")
print(importance.to_string(index=False))
