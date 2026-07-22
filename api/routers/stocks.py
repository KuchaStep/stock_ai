from fastapi import APIRouter, HTTPException

from api.services.stock_service import (
    get_stock_list,
    get_stock
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