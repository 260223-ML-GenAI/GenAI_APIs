from fastapi import FastAPI

from app.routers import langchain_router, vectordb_router

# Set up our FastAPI instance
app = FastAPI()

# Register routers
app.include_router(langchain_router.router)
app.include_router(vectordb_router.router)