import sqlite3
import pandas as pd

conn = sqlite3.connect("database/stock.db")

print(pd.read_sql("PRAGMA table_info(dividends)", conn))

conn.close()