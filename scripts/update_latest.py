"""
update_latest.py
DBの最終日付を確認し、差分の株価・財務・配当データのみを取得・追記する差分更新スクリプト。
"""
import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
import time
from datetime import timedelta

DB_PATH = "database/stock.db"

conn = sqlite3.connect(DB_PATH)
stocks = pd.read_sql("SELECT * FROM stocks", conn)

# --- DBの最終日付を確認 ---
latest_date_str = pd.read_sql("SELECT MAX(date) as max_date FROM prices", conn)["max_date"].iloc[0]
if latest_date_str is None:
    print("[ERROR] prices table is empty. Please run collect_all_stocks_v2.py first.")
    conn.close()
    exit(1)

# 最終日付の翌日から取得開始
latest_date = pd.to_datetime(latest_date_str).tz_localize(None)
start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
today_str = pd.Timestamp.today().strftime("%Y-%m-%d")

print(f"DB last date : {latest_date.strftime('%Y-%m-%d')}")
print(f"Fetching from: {start_date} to {today_str}")
print()

if start_date >= today_str:
    print("Already up to date. No new data to fetch.")
    conn.close()
    exit(0)

new_prices = []
new_fundamentals = []
new_dividends = []

for _, row in stocks.iterrows():
    code = row["code"]
    ticker = f"{code}.T"
    print(f"Updating: {ticker}")

    t_obj = yf.Ticker(ticker)

    # 1. 差分の株価データ取得（最終日翌日〜今日）
    df_price = t_obj.history(start=start_date)
    if not df_price.empty:
        df_price = df_price.reset_index()
        df_price.columns = [c.lower() for c in df_price.columns]
        df_price["code"] = code
        new_prices.append(df_price[["code", "date", "open", "high", "low", "close", "volume"]])
        print(f"  Prices: {len(df_price)} new rows")
    else:
        print(f"  Prices: no new data")

    # 2. 財務データの更新（直近の四半期情報を再取得してUPSERT）
    try:
        q_fin = t_obj.quarterly_financials.T
        q_bal = t_obj.quarterly_balance_sheet.T

        if not q_fin.empty:
            if not q_bal.empty:
                q_all = pd.concat([q_fin, q_bal], axis=1, sort=False)
                q_all = q_all.loc[:, ~q_all.columns.duplicated()]
            else:
                q_all = q_fin.copy()

            q_all = q_all.reset_index()
            q_all = q_all.rename(columns={"index": "announcement_date"})
            q_all["code"] = code

            if "Basic EPS" not in q_all.columns:
                q_all["Basic EPS"] = np.nan

            shares = q_all.get("Basic Average Shares") if "Basic Average Shares" in q_all.columns else q_all.get("Diluted Average Shares")
            if shares is not None and "Net Income" in q_all.columns:
                q_all["Basic EPS"] = q_all["Basic EPS"].fillna(q_all["Net Income"] / shares)

            if "BPS" not in q_all.columns:
                q_all["BPS"] = np.nan
            if shares is not None and "Stockholders Equity" in q_all.columns:
                q_all["BPS"] = q_all["Stockholders Equity"] / shares
            elif shares is not None and "Common Stock Equity" in q_all.columns:
                q_all["BPS"] = q_all["Common Stock Equity"] / shares

            desired_cols = [
                "code", "announcement_date", "Total Revenue", "Operating Income",
                "Net Income", "Basic EPS", "BPS", "Stockholders Equity", "Total Assets"
            ]
            for col in desired_cols:
                if col not in q_all.columns:
                    q_all[col] = np.nan

            new_fundamentals.append(q_all[desired_cols])
    except Exception as e:
        print(f"  Fundamentals fetch failed ({ticker}): {e}")

    # 3. 配当データの更新（全履歴を再取得してUPSERT）
    try:
        divs = t_obj.dividends
        if not divs.empty:
            df_div = divs.reset_index()
            df_div.columns = ["date", "dividend"]
            df_div["code"] = code
            df_div["date"] = df_div["date"].dt.strftime("%Y-%m-%d")
            new_dividends.append(df_div[["code", "date", "dividend"]])
    except Exception as e:
        print(f"  Dividends fetch failed ({ticker}): {e}")

    time.sleep(0.5)  # API制限回避のウェイト（差分なので短めに設定）

print()
print("Saving to database...")

# 新しい株価データをDBに追記 (INSERT OR REPLACE で重複回避)
if new_prices:
    result_prices = pd.concat(new_prices, ignore_index=True)
    # 日付をtz-naiveな文字列に変換して保存
    result_prices["date"] = pd.to_datetime(result_prices["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
    result_prices.to_sql("prices_temp", conn, if_exists="replace", index=False)
    conn.execute("""
        INSERT OR REPLACE INTO prices (code, date, open, high, low, close, volume)
        SELECT code, date, open, high, low, close, volume FROM prices_temp
    """)
    conn.execute("DROP TABLE prices_temp")
    conn.commit()
    print(f"  Prices saved: {len(result_prices)} rows")
else:
    print("  Prices: nothing to save")

# 財務データは全件REPLACE（常に最新の四半期情報で上書き）
if new_fundamentals:
    result_fundamentals = pd.concat(new_fundamentals, ignore_index=True)
    result_fundamentals.to_sql("fundamentals", conn, if_exists="replace", index=False)
    print(f"  Fundamentals saved: {len(result_fundamentals)} rows")

# 配当データは全件REPLACE（最新の配当履歴で上書き）
if new_dividends:
    result_dividends = pd.concat(new_dividends, ignore_index=True)
    result_dividends.to_sql("dividends", conn, if_exists="replace", index=False)
    print(f"  Dividends saved: {len(result_dividends)} rows")

conn.close()
print()
print("Update completed!")
