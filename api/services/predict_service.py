import sqlite3

import joblib
import pandas as pd

from scripts.feature_utils import FEATURES_V4, add_v4_features


DB_PATH = "database/stock.db"
MODEL_PATH = "data/models/stock_model_v4.pkl"


def get_ranking():

    model = joblib.load(MODEL_PATH)

    conn = sqlite3.connect(DB_PATH)

    prices = pd.read_sql(
        "SELECT * FROM prices ORDER BY code, date",
        conn,
    )

    fundamentals = pd.read_sql(
        "SELECT * FROM fundamentals ORDER BY code, announcement_date",
        conn,
    )

    stocks = pd.read_sql(
        "SELECT * FROM stocks",
        conn,
    )
    
    dividends = pd.read_sql(
        "SELECT * FROM dividends ORDER BY code, date",
        conn,
    )

    conn.close()

    features_df = add_v4_features(
        prices,
        fundamentals,
        dividends,
    )

    latest = (
        features_df
        .sort_values("date")
        .groupby("code")
        .tail(1)
    )

    latest = latest.dropna(
        subset=[
            "MA25_ratio",
            "volatility_25",
        ]
    )

    latest["up_probability"] = (
        model.predict_proba(
            latest[FEATURES_V4]
        )[:, 1] * 100
    )

    latest["code"] = latest["code"].astype(str)
    stocks["code"] = stocks["code"].astype(str)

    latest = latest.merge(
        stocks[
            [
                "code",
                "name"
            ]
        ],
        on="code",
        how="left"
    )

    ranking = (
        latest[
            [
                "code",
                "name",
                "date",
                "close",
                "up_probability"
            ]
        ]
        .sort_values(
            "up_probability",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return ranking