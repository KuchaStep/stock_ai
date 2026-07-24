from fastapi import APIRouter
from api.services.update_service import update_data

router = APIRouter()

@router.post("/update")
def update():

    return update_data()