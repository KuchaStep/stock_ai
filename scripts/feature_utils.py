import pandas as pd
import numpy as np

FEATURES_V4 = [
    # 財務（成長率と利益率に置き換え）
    "total_revenue_YoY_growth",
    "operating_income_YoY_growth",
    "net_income_YoY_growth",
    "operating_margin",
    "net_profit_margin",
    "eps",
    "PER",
    "PBR",
    "ROE",
    "ROA",
    "equity_ratio",
    "dividend_yield",
    # テクニカル位置（既存・追加）
    "MA25_ratio",
    "MA75_ratio",
    "volatility_25",
    "RSI",
    "high52w_gap",
    # 勢い・資金流入
    "return",
    "return_vs_market",
    "volume_zscore",
    # ローソク足パターン
    "upper_shadow",
    "lower_shadow",
    "body_ratio",
]


def add_v4_features(prices: pd.DataFrame, fundamentals: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """Create point-in-time v4 features from price and fundamentals data."""
    prices = prices.copy()
    prices = prices.dropna(subset=["close"])
    fundamentals = fundamentals.copy()
    dividends = dividends.copy()

    # 大文字・スペース表記のカラム名を小文字スネークケースに変換
    rename_dict = {
        "Total Revenue": "total_revenue",
        "Operating Income": "operating_income",
        "Net Income": "net_income",
        "Basic EPS": "eps",
        "BPS": "bps",
        "Stockholders Equity": "stockholders_equity",
        "Total Assets": "total_assets",
    }
    fundamentals = fundamentals.rename(columns=rename_dict)

    prices["code"] = prices["code"].astype(str)
    fundamentals["code"] = fundamentals["code"].astype(str)
    prices["date"] = pd.to_datetime(prices["date"].astype(str).str.replace(r"\+.*$", "", regex=True))
    fundamentals["announcement_date"] = pd.to_datetime(
        fundamentals["announcement_date"].astype(str).str.replace(r"\+.*$", "", regex=True)
    )

    # 財務データをコードと発表日でソート
    fundamentals = fundamentals.sort_values(["code", "announcement_date"]).reset_index(drop=True)

    # --- 財務指標の正規化と成長率計算 ---
    # 売上高成長率 (YoY: 前年同期比)
    # `periods=4`は四半期データで前年同期を指す
    fundamentals["total_revenue_YoY_growth"] = fundamentals.groupby("code")[
        "total_revenue"
    ].pct_change(periods=4)

    # 営業利益成長率 (YoY)
    fundamentals["operating_income_YoY_growth"] = fundamentals.groupby("code")[
        "operating_income"
    ].pct_change(periods=4)

    # 純利益成長率 (YoY)
    fundamentals["net_income_YoY_growth"] = fundamentals.groupby("code")[
        "net_income"
    ].pct_change(periods=4)

    # 営業利益率
    # 売上高が0の場合の除算エラーを避けるため、0をNaNに置き換え
    fundamentals["operating_margin"] = fundamentals["operating_income"] / fundamentals["total_revenue"].replace(0, np.nan)

    # 純利益率
    fundamentals["net_profit_margin"] = fundamentals["net_income"] / fundamentals["total_revenue"].replace(0, np.nan)

    # ROE, ROA, 自己資本比率 (equity_ratio) の計算
    eq = fundamentals["stockholders_equity"].replace(0, np.nan)
    assets = fundamentals["total_assets"].replace(0, np.nan)
    
    fundamentals["ROE"] = fundamentals["net_income"] / eq
    fundamentals["ROA"] = fundamentals["net_income"] / assets
    fundamentals["equity_ratio"] = fundamentals["stockholders_equity"] / assets

    # マージに必要な財務特徴量を選択
    fundamental_cols_to_merge = [
        "code", "announcement_date",
        "total_revenue_YoY_growth", "operating_income_YoY_growth", "net_income_YoY_growth",
        "operating_margin", "net_profit_margin",
        "eps", "bps", "ROE", "ROA", "equity_ratio"
    ]
    all_features = []

    for code, price_df in prices.groupby("code"):
        fund_df = fundamentals[fundamentals["code"] == code].copy()
        if fund_df.empty:
            continue

        price_df = price_df.sort_values("date").copy()
        
        # 財務データに必要なカラムがあるか確認して抽出
        avail_merge_cols = [c for c in fundamental_cols_to_merge if c in fund_df.columns]
        fund_df = fund_df.sort_values("announcement_date")[avail_merge_cols].copy() # 処理済みの財務データを使用

        merged_df = pd.merge_asof(
            price_df,
            fund_df,
            left_on="date",
            right_on="announcement_date",
            direction="backward",
        )

        if "code_x" in merged_df.columns:
            merged_df = merged_df.rename(columns={"code_x": "code"})
        if "code_y" in merged_df.columns:
            merged_df = merged_df.drop(columns=["code_y"])

        # 移動平均乖離率
        merged_df["MA25_ratio"] = (
            merged_df["close"] / merged_df["close"].rolling(25).mean() - 1
        )
        merged_df["MA75_ratio"] = (
            merged_df["close"] / merged_df["close"].rolling(75).mean() - 1
        )
        
        # PER & PBR の計算 (0以下はNaNにする)
        if "eps" in merged_df.columns:
            merged_df["PER"] = merged_df["close"] / merged_df["eps"]
            merged_df.loc[merged_df["eps"] <= 0, "PER"] = np.nan
        else:
            merged_df["PER"] = np.nan
            
        if "bps" in merged_df.columns:
            merged_df["PBR"] = merged_df["close"] / merged_df["bps"]
            merged_df.loc[merged_df["bps"] <= 0, "PBR"] = np.nan
        else:
            merged_df["PBR"] = np.nan
            
        # 配当データの処理と配当利回りの計算
        stock_divs = dividends[dividends["code"] == code].copy()
        if not stock_divs.empty:
            stock_divs["date"] = pd.to_datetime(stock_divs["date"]).dt.tz_localize(None)
            stock_divs = stock_divs.sort_values("date")
            
            # カレンダー全日のレンジを作成
            all_days = pd.date_range(start=price_df['date'].min(), end=price_df['date'].max(), freq='D')
            
            # 同じ日に複数の配当がある可能性に備えて集計し、カレンダー日付でリインデックス
            daily_divs = stock_divs.groupby('date')['dividend'].sum().reindex(all_days, fill_value=0.0)
            
            # 365日のローリング和（過去1年間の合計配当）を計算
            ttm_div = daily_divs.rolling(window=365, min_periods=1).sum()
            ttm_div_df = pd.DataFrame({'date': ttm_div.index, 'ttm_dividend': ttm_div.values})
            
            # dailyの株価データにマージ
            merged_df = pd.merge(merged_df, ttm_div_df, on='date', how='left')
            merged_df['dividend_yield'] = merged_df['ttm_dividend'] / merged_df['close']
            merged_df['dividend_yield'] = merged_df['dividend_yield'].fillna(0.0)
            merged_df = merged_df.drop(columns=['ttm_dividend'])
        else:
            merged_df['dividend_yield'] = 0.0
        
        # ボラティリティ
        merged_df["volatility_25"] = merged_df["close"].pct_change().rolling(25).std()

        # RSI
        delta = merged_df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        merged_df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

        # 52週高値からの距離
        merged_df["high52w_gap"] = merged_df["close"] / merged_df["high"].rolling(252).max() - 1

        # リターンと出来高
        merged_df["return"] = merged_df["close"].pct_change()
        vol_mean = merged_df["volume"].rolling(20).mean()
        vol_std = merged_df["volume"].rolling(20).std()
        merged_df["volume_zscore"] = (merged_df["volume"] - vol_mean) / (vol_std + 1e-9)

        # ローソク足パターン
        candle_range = (merged_df["high"] - merged_df["low"]).replace(0, np.nan)
        merged_df["upper_shadow"] = (merged_df["high"] - merged_df[["open", "close"]].max(axis=1)) / candle_range
        merged_df["lower_shadow"] = (merged_df[["open", "close"]].min(axis=1) - merged_df["low"]) / candle_range
        merged_df["body_ratio"] = (merged_df["close"] - merged_df["open"]).abs() / candle_range

        all_features.append(merged_df)

    if not all_features:
        return pd.DataFrame(columns=list(prices.columns) + FEATURES_V4)

    features_df = pd.concat(all_features, ignore_index=True)

    # 市場平均との乖離（セクターや全体相場の影響を排除）
    market_avg = (
        features_df.groupby("date")["return"]
        .mean()
        .rename("market_return")
    )
    features_df = features_df.merge(market_avg, on="date", how="left")
    features_df["return_vs_market"] = features_df["return"] - features_df["market_return"]

    return features_df.sort_values(["code", "date"]).reset_index(drop=True)
