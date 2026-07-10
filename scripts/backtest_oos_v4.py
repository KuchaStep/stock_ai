import sqlite3

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
    conn.close()

    model = joblib.load(MODEL_PATH)

    df = df.dropna(subset=["future_return_60"]).sort_values("date")

    unique_dates = sorted(df["date"].unique())
    split_index = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_index]
    test_df = df[df["date"] >= split_date].copy()

    test_df["score"] = model.predict_proba(test_df[FEATURES_V4])[:, 1]

    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    portfolio_history = []

    test_dates = sorted(test_df["date"].unique())
    for date in test_dates[::REBALANCE_INTERVAL]:
        day_data = test_df[test_df["date"] == date]
        top = day_data.sort_values("score", ascending=False).head(TOP_N)

        if len(top) < TOP_N:
            continue

        period_return = top["future_return_60"].mean()
        net_return = period_return - (2 * TRANSACTION_COST)
        capital *= 1 + net_return
        equity_curve.append(capital)
        portfolio_history.append(
            {
                "date": date,
                "return": net_return,
                "codes": ",".join(top["code"].astype(str).tolist()),
            }
        )

    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdowns = (equity_series - rolling_max) / rolling_max
    history_df = pd.DataFrame(portfolio_history)

    final_return = (capital / INITIAL_CAPITAL - 1) * 100
    max_drawdown = drawdowns.min() * 100
    win_rate = (history_df["return"] > 0).mean() * 100 if not history_df.empty else np.nan

    print("===== V4 OOS backtest =====")
    print(f"Test period      : {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"Top N            : {TOP_N}")
    print(f"Rebalance days   : {REBALANCE_INTERVAL}")
    print(f"Rebalances       : {len(history_df)}")
    print(f"Initial capital  : {INITIAL_CAPITAL:,.0f}")
    print(f"Final capital    : {capital:,.0f}")
    print(f"Total return     : {final_return:.2f} %")
    print(f"Max drawdown     : {max_drawdown:.2f} %")
    print(f"Win rate         : {win_rate:.2f} %")

    if not history_df.empty:
        print("\nRecent portfolios")
        print(history_df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
