import sqlite3
import pandas as pd

conn = sqlite3.connect(
    "database/stock.db"
)

df = pd.read_sql(
    """
    SELECT
        code,
        COUNT(*) as cnt
    FROM prices
    GROUP BY code
    ORDER BY code
    """,
    conn
)

print(df)

conn.close()