from fastapi import FastAPI

from api.routers.stocks import router as stocks_router
from api.routers.ranking import router as ranking_router
from api.routers import backtest

app = FastAPI(
    title="Stock AI API",
    version="1.0.0"
)

app.include_router(stocks_router)
app.include_router(ranking_router)
app.include_router(backtest.router)

@app.get("/")
def root():
    return {
        "status": "ok"
    }