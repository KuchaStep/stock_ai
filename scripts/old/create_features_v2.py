import sqlite3
import pandas as pd
import numpy as np

# =======================
# 追加特徴量
# ① 52週高値・安値からの距離 (high52w_gap, low52w_gap)
# ② 出来高偏差 (volume_zscore)
# ③ ボリンジャーバンド %B (bb_pct)
# ④ ATR (atr) — 値動きの荒さ
# ⑤ 騰落率偏差 (return_vs_market) — 市場平均との差
# ⑥ 価格モメンタム複数期間 (mom3, mom10, mom20)
# =======================

conn = sqlite3.connect(
    "database/stock.db"
)

prices = pd.read_sql(
    """
    SELECT *
    FROM prices
    ORDER BY code, date
    """,
    conn
)

all_features = []

for code, df in prices.groupby("code"):

    print(f"処理中: {code}")

    df = df.copy().reset_index(drop=True)

    # -----------------------
    # 既存特徴量（変更なし）
    # -----------------------
    df["MA5"]  = df["close"].rolling(5).mean()
    df["MA25"] = df["close"].rolling(25).mean()
    df["MA75"] = df["close"].rolling(75).mean()

    delta    = df["close"].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["return"]        = df["close"].pct_change()
    df["volume_change"] = df["volume"].pct_change()
    df["ma5_gap"]       = df["close"] / df["MA5"]  - 1
    df["ma25_gap"]      = df["close"] / df["MA25"] - 1
    df["ma75_gap"]      = df["close"] / df["MA75"] - 1

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["volatility"]  = df["return"].rolling(20).std()

    # -----------------------
    # ① 52週高値・安値からの距離
    # 高値ブレイクアウト候補を検出
    # -----------------------
    df["high52w"] = df["high"].rolling(252).max()
    df["low52w"]  = df["low"].rolling(252).min()
    df["high52w_gap"] = df["close"] / df["high52w"] - 1  # 0に近いほど高値圏
    df["low52w_gap"]  = df["close"] / df["low52w"]  - 1  # 大きいほど安値から回復

    # -----------------------
    # ② 出来高Zスコア
    # 直近20日平均に対して今日の出来高が何σか
    # 急騰/急落の先行指標になりやすい
    # -----------------------
    vol_mean = df["volume"].rolling(20).mean()
    vol_std  = df["volume"].rolling(20).std()
    df["volume_zscore"] = (df["volume"] - vol_mean) / (vol_std + 1e-9)

    # -----------------------
    # ③ ボリンジャーバンド %B
    # 0以下=下限割れ、1以上=上限超え
    # 平均回帰・ブレイクアウト両方で使える
    # -----------------------
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_pct"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # -----------------------
    # ④ ATR (Average True Range)
    # 値動きの絶対的な大きさ → 正規化してボラ比較に
    # -----------------------
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close  = (df["low"]  - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr14      = true_range.rolling(14).mean()
    df["atr_pct"] = atr14 / df["close"]  # 株価比で正規化

    # -----------------------
    # ⑤ 価格モメンタム（複数期間）
    # 3日・10日・20日前からのリターン
    # 短期モメンタムと中期モメンタムの乖離も重要
    # -----------------------
    df["mom3"]  = df["close"] / df["close"].shift(3)  - 1
    df["mom10"] = df["close"] / df["close"].shift(10) - 1
    df["mom20"] = df["close"] / df["close"].shift(20) - 1

    all_features.append(df)

features_df = pd.concat(
    all_features,
    ignore_index=True
)

# -----------------------
# ⑥ 騰落率偏差（市場平均との差）
# 同じ日の全銘柄平均リターンを引く
# 「市場全体が上がった」の影響を除去
# -----------------------
market_avg = (
    features_df
    .groupby("date")["return"]
    .mean()
    .rename("market_return")
)

features_df = features_df.merge(
    market_avg,
    on="date",
    how="left"
)

features_df["return_vs_market"] = (
    features_df["return"] - features_df["market_return"]
)

# -----------------------
# 保存
# -----------------------
features_df.to_sql(
    "features_v2",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print()
print("===== 新特徴量 サンプル確認 =====")
new_cols = [
    "high52w_gap", "low52w_gap",
    "volume_zscore", "bb_pct", "atr_pct",
    "mom3", "mom10", "mom20",
    "return_vs_market"
]
print(features_df[new_cols].describe().round(4))
print()
print("features_v2 作成完了")
print(f"総行数: {len(features_df):,}")
