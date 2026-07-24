from fastapi import APIRouter

from api.services.train_service import train_model

router = APIRouter()


@router.post("/train")
def train():

    return train_model()