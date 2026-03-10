from typing import Any

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# This Service is full of functions that let our VectorDB work
# Creating/Getting collections, ingesting data as vector embeddings, and searching the data

PERSIST_DIRECTORY = "app/chroma_db" # Where the DB lives in our application

# The model we pulled - this is the AI model that turns our data into vector embeddings
EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")

# The Chroma store itself (which contains the collections)
vector_store: dict[str, Chroma] = {}

# Create a new collection in VectorDB - a collection in VDB is like a table in SQL
def create_collection(collection_name:str):

    # If the Collection doesn't already exist, make a new one
    if collection_name not in vector_store:
        vector_store[collection_name] = Chroma(
            collection_name = collection_name,
            persist_directory = PERSIST_DIRECTORY,
            embedding_function = EMBEDDING_MODEL
        )

    return vector_store[collection_name]


# Ingest JSON items into the DB
def ingest_json(collection_name:str, items:list[dict[str, Any]]):

    # Get an instance of the collection in the vector store

    # Turn the list of items into a list of document to get ingested

    # Attach IDs to each document (easy cuz it's the sample data)

    # Add the documents, perform the ingestion, and we have embedded vectors!


# TODO: ingest_text


# TODO: search_collection (can be used for both of our collections!)
