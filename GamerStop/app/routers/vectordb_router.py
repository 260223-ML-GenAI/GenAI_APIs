from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.langchain_service import get_basic_chain
from app.services.vectordb_service import ingest_json, search_collection

router = APIRouter(
    prefix="/vectordb",
    tags=["vectordb"]
)

# Pydantic Model for ingesting JSON items
class IngestItem(BaseModel):
    id: str
    text: str

# Import our basic chain here
basic_chain = get_basic_chain()

# Ingest JSON endpoint
@router.post("/ingest-json")
async def ingest_json_items(items:list[IngestItem]):

    # Call the service method that ingests JSON
    ingested_items = ingest_json(
        collection_name="video_games",
        items=[item.model_dump() for item in items]
    )

    return {"ingested_items": ingested_items}


# Similarity search for video games
# (not RAG, just getting similarity search results directly)
@router.post("/search-games")
async def search_games(query:str, k:int=5):

    # Call search collection from the service and return it
    results = search_collection(
        collection_name="video_games",
        query=query,
        k=k
    )

    return results

# RAG endpoint - get the JSON search results (video_games collection)
# And Augment the LLM's Generated response based on the Retrieved data
@router.get("/games-rag")
async def games_rag(query:str):

    # Get the search results from the vector DB
    results = search_collection(query=query, k=5, collection_name="video_games")

    return basic_chain.invoke(
        input=f"Respond to the User's Query based on the following Search Results. "
              f"ONLY use the information in the Search Results. "
              f"Don't fall back to any outside information, say you don't know if you have to."
                f"User's Query: {query}"
                f"Search Results: {results}")