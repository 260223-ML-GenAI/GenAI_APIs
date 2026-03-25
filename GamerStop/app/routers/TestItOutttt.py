from fastapi import APIRouter

from app.services.TestItOut import build_crew

router = APIRouter(
    prefix="/crew",
    tags=["crew"]
)

@router.post("/recommend")
def crew_recommend(query: str):
    crew = build_crew(query)
    result = crew.kickoff()
    return {"recommendation": result.raw}