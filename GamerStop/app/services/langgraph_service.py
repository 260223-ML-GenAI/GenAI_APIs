from typing import TypedDict

from langchain_ollama import ChatOllama

# This service will define the State, Nodes, and overall Graph
# This is a GRAPH-BASED workflow with LangGraph (as opposed to chain-based with LangChain)

# First, define the LLM like before
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.1
)

# Define the State Object for the Graph - stores temp data we want to keep track of
# Each Node in the Graph can read/write to this state object
class GraphState(TypedDict, total=False): # total=False makes the fields optional
    query:str # The user's query to the LLM
    answer:str # The LLM's answer to the query
    docs:list[dict] # Retrieved documents from a VectorDB search
    route:str # The "routing" decision we make. Determines which Node executes next
    # TODO: memory manager

# ======================(Node Definitions)============================= #

# Each Node is like a "step" in the Graph. Functions that may or may not be invoked
# Nodes have read/write access to the GraphState

# Our first Node - kind of like a Front Controller - The ROUTING NODE
# When the user sends a query, this node decides which path to take
def route_node(state:GraphState) -> GraphState:

    # Get the user's query from state - default to empty string if not present
    query = state.get("query", "").lower()

    # VERY simple keyword matching (FOR NOW) to determine the route
    if any(word in query for word in ["recommend", "recs", "suggest"]):
        return {"route": "recs"}
    elif any(word in query for word in ["review", "critic", "critics"]):
        return {"route": "reviews"}

    # TODO: general chat fallback if no keywords detected