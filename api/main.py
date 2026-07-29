from fastapi import FastAPI

from api.routers.stocks import router as stocks_router
from api.routers.ranking import router as ranking_router
from api.routers import backtest
from api.routers import train
from api.routers import update

from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(
    title="Stock AI API",
    version="1.0.0"
)

app.include_router(stocks_router)
app.include_router(ranking_router)
app.include_router(backtest.router)
app.include_router(update.router)
app.include_router(train.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5000",
        "http://localhost:8000",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "ok"
    }