from fastapi import APIRouter
from langchain_community.document_loaders import TextLoader
from pydantic import BaseModel

from app.services.langchain_service import get_basic_chain, get_sequential_chain

# Router setup
router = APIRouter(
    prefix="/langchain", # Prefix defines which HTTP URLs hit our router
    tags=["langchain"] # These endpoints will be under the "langchain" tag in the SwaggerUI
)

# Quick Pydantic model for the user's query to our LLM
# This helps it play nice with FastAPI (user query goes in the body, not query params)
class ChatRequest(BaseModel):
    input: str

# Import the different chains we'll invoke in our routes
basic_chain = get_basic_chain()
sequential_chain = get_sequential_chain()

# General chat endpoint - no memory or fancy features
@router.post("/chat")
async def general_chat(chat:ChatRequest):

    # Invoke the chain, returning the response
    return basic_chain.invoke(input=chat.input)

# DOCUMENT LOADER - loading in a .txt file to summarize
@router.get("/summarize")
async def summarize():

    # Use a LangChain Document Loader called TextLoader
    loader = TextLoader("app/StarWarsWriteup.txt")

    # Extract the text (comes back as a List of LangChain Document objects)
    document = loader.load()
    text = document[0].page_content # Get the text content as a python String

    # Our first look at RAG -
    # the data we got from the .txt file will help the LLM generate its response
    # Invoke the same basic chain, this time the query will go straight into it
    return basic_chain.invoke(input={f"Summarize this text: {text}"})

# INVOKING OUR SEQUENTIAL CHAIN
@router.post("/support-chat")
async def support_chat(chat:ChatRequest):

    # Almost identical to the first chat endpoint. Just a different chain.
    return sequential_chain.invoke(input=chat.input)