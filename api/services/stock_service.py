from api.db import get_connection
from datetime import datetime, timedelta


def get_stock_list():
    
    conn = get_connection()

    sql = """
        SELECT
            code,
            name
        FROM stocks
        ORDER BY code
    """

    rows = conn.execute(sql).fetchall()

    conn.close()

    return [dict(row) for row in rows]

def get_stock(code: str):

    conn = get_connection()

    sql = """
        SELECT
            code,
            name
        FROM stocks
        WHERE code = ?
    """

    row = conn.execute(sql, (code,)).fetchone()

    conn.close()

    if row is None:
        return None

    return dict(row)

def get_price_history(
    code: str,
    days: int = 365,
    from_date: str | None = None,
    to_date: str | None = None,
    ):
    
    if from_date:
        from_date = datetime.strptime(from_date, "%Y%m%d").strftime("%Y-%m-%d")

    if to_date:
        to_date = datetime.strptime(to_date, "%Y%m%d").strftime("%Y-%m-%d")

    conn = get_connection()

    sql = """
        SELECT
            date,
            open,
            high,
            low,
            close,
            volume
        FROM prices
        WHERE code = ?
    """
    
    params = [code]

    if from_date and to_date:
        sql += " AND DATE(date) BETWEEN DATE(?) AND DATE(?)"
        params.extend([from_date, to_date])

    else:
        start_date = (
            datetime.today() - timedelta(days=days)
        ).strftime("%Y-%m-%d")

        sql += " AND DATE(date) >= DATE(?)"
        params.append(start_date)

    sql += " ORDER BY date"

    rows = conn.execute(sql, params).fetchall()

    conn.close()

    return [dict(row) for row in rows]