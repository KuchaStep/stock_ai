import sqlite3
import pandas as pd

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

    df = df.copy()

    df["MA5"] = (
        df["close"]
        .rolling(5)
        .mean()
    )

    df["MA25"] = (
        df["close"]
        .rolling(25)
        .mean()
    )

    df["MA75"] = (
        df["close"]
        .rolling(75)
        .mean()
    )
    
    delta = df["close"].diff()

    gain = delta.where(
        delta > 0,
        0
    )

    loss = -delta.where(
        delta < 0,
        0
    )

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = (
        100 - (100 / (1 + rs))
    )
    df["return"] = (
        df["close"]
        .pct_change()
    )

    df["volume_change"] = (
        df["volume"]
        .pct_change()
    )

    df["ma5_gap"] = (
        df["close"]
        / df["MA5"]
        - 1
    )

    df["ma25_gap"] = (
        df["close"]
        / df["MA25"]
        - 1
    )

    df["ma75_gap"] = (
        df["close"]
        / df["MA75"]
        - 1
    )
    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = (
        ema12 - ema26
    )

    df["MACD_signal"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )
    
    df["volatility"] = (
        df["return"]
        .rolling(20)
        .std()
    )

    all_features.append(df)


features = pd.concat(
    all_features,
    ignore_index=True
    )


features.to_sql(
    "features_all",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print(
    features.head()
)

print(
    "\nfeatures_all作成完了"
)