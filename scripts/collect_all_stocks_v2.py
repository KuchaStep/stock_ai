import sqlite3
import pandas as pd
import yfinance as yf
import time

conn = sqlite3.connect("database/stock.db")
stocks = pd.read_sql("SELECT * FROM stocks", conn)

all_prices = []
all_fundamentals = []
all_dividends = []

for _, row in stocks.iterrows():
    code = row["code"]
    ticker = f"{code}.T"
    print(f"取得中: {ticker}")
    
    t_obj = yf.Ticker(ticker)
    
    # 1. 株価データの取得（既存ロジック）
    df_price = t_obj.history(start="2010-01-01")
    if len(df_price) == 0:
        continue
        
    df_price = df_price.reset_index()
    df_price.columns = [c.lower() for c in df_price.columns] # 小文字に統一
    df_price["code"] = code
    all_prices.append(df_price[["code", "date", "open", "high", "low", "close", "volume"]])
    
    # 2. 【新規】四半期決算データ（損益計算書と貸借対照表）の取得
    try:
        q_fin = t_obj.quarterly_financials.T
        q_bal = t_obj.quarterly_balance_sheet.T
        
        if not q_fin.empty:
            import numpy as np
            if not q_bal.empty:
                q_all = pd.concat([q_fin, q_bal], axis=1, sort=False)
                q_all = q_all.loc[:, ~q_all.columns.duplicated()]
            else:
                q_all = q_fin.copy()
                
            q_all = q_all.reset_index()
            q_all = q_all.rename(columns={"index": "announcement_date"})
            q_all["code"] = code
            
            # Basic EPS の補正（NaN の場合、Net Income / Basic Average Shares で計算）
            if "Basic EPS" not in q_all.columns:
                q_all["Basic EPS"] = np.nan
            
            shares = q_all.get("Basic Average Shares") if "Basic Average Shares" in q_all.columns else q_all.get("Diluted Average Shares")
            if shares is not None and "Net Income" in q_all.columns:
                q_all["Basic EPS"] = q_all["Basic EPS"].fillna(q_all["Net Income"] / shares)
            
            # BPS の計算
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
                    
            all_fundamentals.append(q_all[desired_cols])
    except Exception as e:
        print(f"財務データ取得失敗 ({ticker}): {e}")
        
    # 3. 【新規】配当履歴データの取得
    try:
        divs = t_obj.dividends
        if not divs.empty:
            df_div = divs.reset_index()
            df_div.columns = ["date", "dividend"]
            df_div["code"] = code
            # 日付を文字列に変換して保存
            df_div["date"] = df_div["date"].dt.strftime("%Y-%m-%d")
            all_dividends.append(df_div[["code", "date", "dividend"]])
    except Exception as e:
        print(f"配当データ取得失敗 ({ticker}): {e}")
        
    time.sleep(1) # API制限回避のためのウェイト

# 株価データの保存
if all_prices:
    result_prices = pd.concat(all_prices, ignore_index=True)
    # 日付をtz-naiveな文字列に変換して保存 (update_latest.pyとの整合性を保つため)
    result_prices["date"] = pd.to_datetime(result_prices["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
    result_prices.to_sql("prices", conn, if_exists="replace", index=False)

# 【新規】財務データの保存
if all_fundamentals:
    result_fundamentals = pd.concat(all_fundamentals, ignore_index=True)
    result_fundamentals.to_sql("fundamentals", conn, if_exists="replace", index=False)

# 【新規】配当データの保存
if all_dividends:
    result_dividends = pd.concat(all_dividends, ignore_index=True)
    result_dividends.to_sql("dividends", conn, if_exists="replace", index=False)

conn.close()
print("\n株価・財務・配当データの取得・保存が完了しました")