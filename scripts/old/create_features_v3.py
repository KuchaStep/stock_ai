import sqlite3
import pandas as pd
import numpy as np

# =======================
# v2からの追加特徴量
#
# 【出来高・資金流入系】
# volume_price_trend  : 価格変動×出来高の累積（資金が流入/流出しているか）
# volume_up_ratio     : 上昇日の出来高比率（買い圧力の強さ）
# volume_spike        : 出来高が直近最大値の何%か（異常出来高の検出）
# obv_slope           : OBV(On Balance Volume)の傾き（資金流入トレンド）
#
# 【価格パターン系】
# gap_up              : 窓開けギャップ（前日終値→当日始値）
# upper_shadow        : 上ヒゲ比率（高値 - max(始値,終値)）/ 値幅
# lower_shadow        : 下ヒゲ比率（min(始値,終値) - 安値）/ 値幅
# body_ratio          : 実体比率（始値〜終値の幅）/ 値幅
# consecutive_up      : 何日連続で上昇しているか
# consecutive_down    : 何日連続で下落しているか
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
    # 既存特徴量（v2と同じ）
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

    df["high52w"]     = df["high"].rolling(252).max()
    df["low52w"]      = df["low"].rolling(252).min()
    df["high52w_gap"] = df["close"] / df["high52w"] - 1
    df["low52w_gap"]  = df["close"] / df["low52w"]  - 1

    vol_mean = df["volume"].rolling(20).mean()
    vol_std  = df["volume"].rolling(20).std()
    df["volume_zscore"] = (df["volume"] - vol_mean) / (vol_std + 1e-9)

    bb_mid   = df["close"].rolling(20).mean()
    bb_std   = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_pct"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close  = (df["low"]  - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_pct"] = true_range.rolling(14).mean() / df["close"]

    df["mom3"]  = df["close"] / df["close"].shift(3)  - 1
    df["mom10"] = df["close"] / df["close"].shift(10) - 1
    df["mom20"] = df["close"] / df["close"].shift(20) - 1

    # -----------------------
    # 【出来高・資金流入系】新規追加
    # -----------------------

    # ① Volume Price Trend (VPT)
    # 価格変化率 × 出来高 を累積
    # プラスなら資金流入、マイナスなら流出
    vpt = (df["return"] * df["volume"]).cumsum()
    vpt_ma = vpt.rolling(10).mean()
    df["vpt_slope"] = (vpt - vpt_ma) / (vpt_ma.abs() + 1e-9)

    # ② 上昇日の出来高比率（直近10日）
    # 上がった日の出来高 / 全出来高
    # 0.5超 = 買い圧力が強い
    up_vol   = df["volume"].where(df["return"] > 0, 0).rolling(10).sum()
    total_vol = df["volume"].rolling(10).sum()
    df["volume_up_ratio"] = up_vol / (total_vol + 1e-9)

    # ③ 出来高スパイク
    # 直近20日の最大出来高に対する今日の出来高比率
    # 1.0 = 過去20日で最大の出来高
    vol_max20 = df["volume"].rolling(20).max()
    df["volume_spike"] = df["volume"] / (vol_max20 + 1e-9)

    # ④ OBV(On Balance Volume)の短期傾き
    # 終値が上昇した日は出来高を加算、下落日は減算
    # 5日前のOBVより増えていれば資金流入トレンド
    obv = (np.sign(df["return"]) * df["volume"]).cumsum()
    obv_ma5 = obv.rolling(5).mean()
    df["obv_slope"] = (obv - obv_ma5) / (df["volume"].rolling(5).mean() + 1e-9)

    # -----------------------
    # 【価格パターン系】新規追加
    # -----------------------

    # ⑤ ギャップ（窓開け）
    # 当日始値 / 前日終値 - 1
    # プラス = 窓開け上昇、マイナス = 窓開け下落
    df["gap_up"] = df["open"] / df["close"].shift(1) - 1

    # ⑥ ローソク足の値幅（分母）
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)

    # ⑦ 上ヒゲ比率
    # 上ヒゲ = 高値 - max(始値, 終値)
    # 長い上ヒゲ = 高値圏で売り圧力が強い（翌日下落示唆）
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    df["upper_shadow"] = upper_wick / candle_range

    # ⑧ 下ヒゲ比率
    # 下ヒゲ = min(始値, 終値) - 安値
    # 長い下ヒゲ = 安値圏で買い支えあり（翌日上昇示唆）
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    df["lower_shadow"] = lower_wick / candle_range

    # ⑨ 実体比率
    # 実体 = |終値 - 始値| / 値幅
    # 大きいほど方向感が明確なローソク足
    body = (df["close"] - df["open"]).abs()
    df["body_ratio"] = body / candle_range

    # ⑩ 連続上昇日数 / 連続下落日数
    # 3日以上連続上昇は過熱感のシグナル
    up   = (df["return"] > 0).astype(int)
    down = (df["return"] < 0).astype(int)

    consec_up   = []
    consec_down = []
    cu = cd = 0

    for u, d in zip(up, down):
        cu = cu + 1 if u == 1 else 0
        cd = cd + 1 if d == 1 else 0
        consec_up.append(cu)
        consec_down.append(cd)

    df["consecutive_up"]   = consec_up
    df["consecutive_down"] = consec_down

    all_features.append(df)

features_df = pd.concat(
    all_features,
    ignore_index=True
)

# -----------------------
# 騰落率偏差（市場平均との差）
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
    "features_v3",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

# 新特徴量の確認
new_cols = [
    "vpt_slope", "volume_up_ratio", "volume_spike", "obv_slope",
    "gap_up", "upper_shadow", "lower_shadow", "body_ratio",
    "consecutive_up", "consecutive_down"
]

print()
print("===== 新特徴量 サンプル確認 =====")
print(features_df[new_cols].describe().round(4))
print()
print("features_v3 作成完了")
print(f"総行数: {len(features_df):,}")
