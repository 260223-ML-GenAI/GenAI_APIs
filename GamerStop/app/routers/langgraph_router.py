from fastapi import APIRouter
from pydantic import BaseModel

from app.services.langgraph_service import graph

router = APIRouter(
    prefix="/langgraph",
    tags=["langgraph"]
)

# Basic Pydantic Model to handle user input
class ChatRequest(BaseModel):
    input: str


# Notice how we just have a single endpoint for 3 potential outcomes
# LangGraph is inherently multi-modal.
# All that complicated Service logic cleans up our routers
@router.post("/langgraph")
async def langgraph_chat(chat:ChatRequest):

    # Get the result of the invocation
    result = graph.invoke({"query":chat.input})

    # Return the result and the route (just to see it)
    return {
        "response": result.get("answer"),
        "route": result.get("route")
    }
