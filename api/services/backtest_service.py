from api.db import get_connection
import pandas as pd

def get_backtest():

    conn = get_connection()

    row = conn.execute("""
        SELECT
            run_date,
            test_days,
            rebalance_interval,
            top_n,
            win_rate,
            avg_return,
            total_return,
            max_drawdown,
            initial_capital,
            final_capital
        FROM backtest_summary
        ORDER BY id DESC
        LIMIT 1
    """).fetchone()

    conn.close()

    if row is None:
        return {}

    return dict(row)


def get_backtest_history():

    conn = get_connection()

    rows = conn.execute("""
        SELECT
            trade_date,
            capital,
            benchmark,
            drawdown
        FROM backtest_history
        ORDER BY trade_date
    """).fetchall()

    conn.close()

    return [dict(row) for row in rows]