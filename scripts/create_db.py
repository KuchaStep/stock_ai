import sqlite3

conn = sqlite3.connect("database/stock.db")
cur = conn.cursor()

# 1. 銘柄マスターテーブル（既存）
cur.execute("""
CREATE TABLE IF NOT EXISTS stocks (
    code TEXT PRIMARY KEY,
    name TEXT,
    market TEXT,
    sector TEXT
)
""")

# 2. 株価ヒストリカルテーブル（既存）
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

# 3. 【新規追加】四半期財務データテーブル
# yfinanceやJ-Quantsから取得する時系列の決算数値を格納します
cur.execute("""
CREATE TABLE IF NOT EXISTS fundamentals (
    code TEXT,
    announcement_date TEXT,    -- 決算発表日（データリーク防止に必須）
    fiscal_quarter TEXT,       -- 対象四半期（例: 2025Q1）
    total_revenue REAL,        -- 売上高
    operating_income REAL,     -- 営業利益
    net_income REAL,           -- 純利益
    eps REAL,                  -- 1株当たり利益（yfinance等から取得可能な場合）
    PRIMARY KEY(code, announcement_date)
)
""")

##バッグテスト用のデータテーブル
cur.execute("""
CREATE TABLE IF NOT EXISTS backtest_summary (
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

cur.execute("""
CREATE TABLE IF NOT EXISTS backtest_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT,
    trade_date TEXT,
    capital REAL,
    benchmark REAL,
    drawdown REAL
)
""")

# 4. 【新規追加】データ結合を高速化するためのインデックス
# codeとdateによる検索・結合クエリのパフォーマンスを最大化します
cur.execute("CREATE INDEX IF NOT EXISTS idx_prices_code_date ON prices(code, date);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_fundamentals_code_date ON fundamentals(code, announcement_date);")

conn.commit()
conn.close()

print("中長期AI向けへのDB拡張・インデックス作成が完了しました。")