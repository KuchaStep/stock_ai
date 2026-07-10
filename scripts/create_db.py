import sqlite3

conn = sqlite3.connect("database/stock.db")

cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS stocks (
    code TEXT PRIMARY KEY,
    name TEXT,
    market TEXT,
    sector TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS prices (
    code TEXT,
    date TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    PRIMARY KEY(code, date)
)
""")

conn.commit()
conn.close()

print("DB作成完了")