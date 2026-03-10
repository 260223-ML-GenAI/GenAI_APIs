from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.vectordb_service import ingest_json

router = APIRouter(
    prefix="/vectordb",
    tags=["vectordb"]
)

# Pydantic Model for ingesting JSON items
class IngestItem(BaseModel):
    id: str
    text: str


# Ingest JSON endpoint
@router.post("/ingest-json")
async def ingest_json_items(items:list[IngestItem]):

    # Call the service method that ingests JSON
    ingested_items = ingest_json(
        collection_name="video_games",
        items=[item.model_dump() for item in items]
    )

    return {"ingested_items": ingested_items}