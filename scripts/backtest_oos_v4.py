import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from feature_utils import FEATURES_V4

DB_PATH = "database/stock.db"
MODEL_PATH = "data/models/stock_model_v4.pkl"

INITIAL_CAPITAL = 1_000_000.0
REBALANCE_INTERVAL = 60
TOP_N = 5
TRANSACTION_COST = 0.0015


def main() -> None:
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("SELECT * FROM training_data_v4", conn)

    model = joblib.load(MODEL_PATH)

    df = df.dropna(subset=["future_return_60"]).sort_values("date")

    unique_dates = sorted(df["date"].unique())
    split_index = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_index]
    test_df = df[df["date"] >= split_date].copy()

    test_df["score"] = model.predict_proba(test_df[FEATURES_V4])[:, 1]

    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    history = []

    for date in sorted(test_df["date"].unique())[::REBALANCE_INTERVAL]:
        day = test_df[test_df["date"] == date]
        top = day.sort_values("score", ascending=False).head(TOP_N)

        if len(top) < TOP_N:
            continue

        period_return = top["future_return_60"].mean()
        net_return = period_return - (2 * TRANSACTION_COST)

        capital *= (1 + net_return)
        equity_curve.append(capital)

        history.append({
            "trade_date": date,
            "capital": capital,
            "benchmark": None,
            "drawdown": None,
            "codes": ",".join(top["code"].astype(str))
        })

    equity = pd.Series(equity_curve)
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max

    final_return = (capital / INITIAL_CAPITAL - 1) * 100
    max_drawdown = drawdown.min() * 100
    history_df = pd.DataFrame(history)
    win_rate = (history_df["capital"].diff() > 0).mean() * 100 if len(history_df) else np.nan
    avg_return = history_df["capital"].pct_change().mean() * 100 if len(history_df) else np.nan

    conn.execute("""
    CREATE TABLE IF NOT EXISTS backtest_summary(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_date TEXT,
        test_days INTEGER,
        rebalance_interval INTEGER,
        top_n INTEGER,
        win_rate REAL,
        avg_return REAL,
        total_return REAL,
        max_drawdown REAL,
        initial_capital REAL,
        final_capital REAL
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS backtest_history(
        trade_date TEXT,
        capital REAL,
        benchmark REAL,
        drawdown REAL
    )
    """)

    conn.execute("DELETE FROM backtest_summary")
    conn.execute("DELETE FROM backtest_history")

    conn.execute("""
    INSERT INTO backtest_summary(
        run_date,test_days,rebalance_interval,top_n,
        win_rate,avg_return,total_return,max_drawdown,
        initial_capital,final_capital
    )
    VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        len(test_df),
        REBALANCE_INTERVAL,
        TOP_N,
        float(win_rate) if not np.isnan(win_rate) else None,
        float(avg_return) if not np.isnan(avg_return) else None,
        final_return,
        max_drawdown,
        INITIAL_CAPITAL,
        capital
    ))

    history_df["benchmark"] = None
    history_df["drawdown"] = drawdown.iloc[1:].values[:len(history_df)]

    history_df[["trade_date","capital","benchmark","drawdown"]].to_sql(
        "backtest_history",
        conn,
        if_exists="append",
        index=False
    )

    conn.commit()
    conn.close()

    print("===== BACKTEST =====")
    print(f"Final Capital : {capital:,.0f}")
    print(f"Total Return  : {final_return:.2f}%")
    print(f"Win Rate      : {win_rate:.2f}%")
    print(f"Max Drawdown  : {max_drawdown:.2f}%")


if __name__ == "__main__":
    main()
