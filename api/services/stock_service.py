from api.db import get_connection


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
