from fastapi import APIRouter

from api.services.backtest_service import (
    get_backtest,
    get_backtest_history,
)

router = APIRouter()


@router.get("/backtest")
def backtest():
    return get_backtest()


@router.get("/backtest/history")
def backtest_history():
    return get_backtest_history()