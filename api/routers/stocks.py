from fastapi import APIRouter, HTTPException

from api.services.stock_service import (
    get_stock_list,
    get_stock,
    get_price_history
)

router = APIRouter(
    prefix="/stocks",
    tags=["Stocks"]
)


@router.get("")
def stocks():
    return get_stock_list()


@router.get("/{code}")
def stock(code: str):

    result = get_stock(code)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Stock not found"
        )

    return result

@router.get("/{code}/history")
def stock_history(
    code: str,
    days: int = 365,
    from_date: str | None = None,
    to_date: str | None = None,
    ):
    return get_price_history(
        code,
        days=days,
        from_date=from_date,
        to_date=to_date,
    )
    