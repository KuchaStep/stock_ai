import sqlite3

import pandas as pd

from feature_utils import add_v4_features


DB_PATH = "database/stock.db"


def main() -> None:
    conn = sqlite3.connect(DB_PATH)

    prices = pd.read_sql(
        "SELECT * FROM prices ORDER BY code, date",
        conn,
    )
    fundamentals = pd.read_sql(
        "SELECT * FROM fundamentals ORDER BY code, announcement_date",
        conn,
    )
    dividends = pd.read_sql(
        "SELECT * FROM dividends ORDER BY code, date",
        conn,
    )

    features_df = add_v4_features(prices, fundamentals, dividends)
    features_df["date"] = features_df["date"].dt.strftime("%Y-%m-%d")
    if "announcement_date" in features_df.columns:
        features_df["announcement_date"] = features_df["announcement_date"].dt.strftime(
            "%Y-%m-%d"
        )

    features_df.to_sql("features_v4", conn, if_exists="replace", index=False)
    conn.close()

    print(f"features_v4 created: {len(features_df):,} rows")


if __name__ == "__main__":
    main()
