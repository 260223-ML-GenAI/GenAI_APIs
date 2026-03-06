from fastapi import APIRouter

# Router setup
router = APIRouter(
    prefix="/langchain", # Prefix defines which HTTP URLs hit our router
    tags=["langchain"] # These endpoints will be under the "langchain" tag in the SwaggerUI
)

# TODO: Quick Pydantic model for the user's query to our LLM

# TODO: Import the different chains we'll invoke in our routes


# General chat endpoint - no memory or fancy features
@router.post("/chat")
async def general_chat():
    return "hello!"