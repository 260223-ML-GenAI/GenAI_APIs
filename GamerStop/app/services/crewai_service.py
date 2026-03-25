from crewai import Agent
from crewai.tools import tool

from app.services.vectordb_service import search_collection


# CrewAI - A framework that lets us define "crews" of Agents
# These Agents work together to accomplish complex multi-step processes.

# Defined in this service:
    # Tools - functions an Agent can use
    # Agents - the entities that performs Tasks
    # Tasks - specific processes assigned to an Agent
    # Crew - the group of Agents we invoke to accomplish a goal


# =========== Defining Tools the Agents can use ===========

# A tool that does a similarity search on the video_games VectorDB collection
@tool("search_games")
def search_games_tool(query:str):
    """This tool performs a similarity search on the video_games VectorDB collection.
    This is primarily for video game recommendations and suggestions.
    The result of this tool is a list of relevant video games for RAG functionality"""
    results = search_collection(collection_name="video_games", query=query, k=5)
    return str(results) # I want it as a String for ease of use

# TODO: A tool that uses the similarity search data to generate a game recommendation


# ============ Defining Agents to perform specialized tasks ==========

# Agent 1 - The Researcher - gathers data from VectorDB
researcher = Agent(
    role="Game Research Specialist",
    goal="Based on the user's query, find relevant game info from VectorDB. "
         "Use only the search_games tool to do your research. "
         "Return the exact data that the tool provides, without adding to it. ",
    backstory="You are a gaming expert that specializes in data gathering. ",
    tools=[search_games_tool],
    max_iter=1 # Only runs once
    # TODO: point to the LLM we want to use (BEN: bedrock?)
)

# TODO: Agent 2 - The Analyst - takes raw data from Researcher and builds a response


# ============ A function that defines Task order, and builds the crew =======




