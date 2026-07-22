from fastapi import APIRouter

from api.services.ranking_service import ranking

router = APIRouter(
    prefix="/ranking",
    tags=["Ranking"]
)


@router.get("")
def get_ranking():
    return ranking()