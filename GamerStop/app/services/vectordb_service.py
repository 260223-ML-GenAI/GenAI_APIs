import hashlib
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# This Service is full of functions that let our VectorDB work
# Creating/Getting collections, ingesting data as vector embeddings, and searching the data

PERSIST_DIRECTORY = "app/chroma_db" # Where the DB lives in our application

# The model we pulled - this is the AI model that turns our data into vector embeddings
EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")

# The Chroma store itself (which contains the collections)
vector_store: dict[str, Chroma] = {}

# Create a new collection in VectorDB - a collection in VDB is like a table in SQL.
# If the collection already exists, just return it.
def create_or_get_collection(collection_name:str):

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
    collection = create_or_get_collection(collection_name)

    # Turn the list of items into a list of document to get ingested
    docs = [
        Document(page_content=item["text"], metadata=item.get("metadata", {}))
        for item in items
    ]

    # Attach IDs to each document (easy cuz it's the sample data)
    ids = [item["id"] for item in items]

    # Add the documents, perform the ingestion, and we have embedded vectors!
    collection.add_documents(docs, ids=ids)

    # Return the length of the ingested items just for visibility
    return len(items)


# Ingest text into the DB (more involved, need to generate our own IDs and chunks)
def ingest_text(collection_name:str, text:str, game_title:str):

    # Basic cleanup - strip whitespace
    text = text.strip()

    # CHUNK the raw text into smaller pieces (using a LangChain Transformer)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, # How many chars should be in each chunk
        chunk_overlap=100, # How many chars overlap between each chunk (captures context)
        separators=["\n\n", "\n", " ", ""] # Preferred split points
    )

    # Get our chunks as a list[str] so we can iterate and reformat
    chunks = splitter.split_text(text)

    # An empty list (will hold the Documents we want to ingest)
    items = []

    # Generate a Document for each chunk, with a unique ID based on text content
    for chunk in enumerate(chunks):

        # Create a hash of the chunk content to ensure uniqueness
        chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()[:8]

        doc_id = f"{game_title}_{chunk_hash}" # Unique ID for each chunk based on title and content
        items.append({
            "id": doc_id,
            "text": chunk,
            "metadata": {"game_title": game_title} # Optional metadata field
        })

# Vector DB Search (can be used for both of our collections!)
def search_collection(collection_name:str, query:str, k:int=5):

    # Get the collection from the vector store
    collection = create_or_get_collection(collection_name)

    # Perform the similarity search
    # (This is a Retriver, we're using it to retrieve data for the LLM's response)
    results = collection.similarity_search_with_score(query, k=k)

    return results