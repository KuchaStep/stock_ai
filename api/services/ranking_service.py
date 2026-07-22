from api.services.predict_service import get_ranking

def ranking():
    return get_ranking().to_dict(orient="records")