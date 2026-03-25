import os
os.environ["RUST_BACKTRACE"] = "full"

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_ollama import ChatOllama

from app.services.langchain_service import get_basic_chain
from app.services.vectordb_service import search_collection

# Import our basic chain and define the LLM
basic_chain = get_basic_chain()

# CrewAI's own LLM wrapper — NOT ChatOllama
llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434" # default Ollama URL
)


# ====== Defining Tools that the Agents will use ======

# A tool that calls the search_collection function for video_games
@tool("search_game_knowledge")
def search_game_knowledge(query: str) -> str:
    """Search for games via VectorDb similarity search.
    Input must be a plain text search string like 'action adventure open world games'.
    Do not pass JSON or dictionaries."""
    # result = search_collection(collection_name="video_games", query=query, k=3)
    # return result

    # Temporary mock data - latest version of chroma doesn't play nice with CrewAI
    return """Zelda Tears of the Kingdom - Action Adventure, open world exploration
    Hollow Knight Silksong - Metroidvania, exploration focused
    Megabonk - Bonk Survival, fast-paced action"""

# A tool that calls the basic chain with a custom prompt to process the search results
@tool("process_game_data")
def process_game_data(game_data: str) -> str:
    """Process raw game data into a user-friendly recommendation format."""
    prompt = f"""You are a helpful assistant that takes raw game data, 
    and turns it into a friendly recommendation for a user. 
    
    Here's the raw data you found: {game_data}. 
    Write a clear, concise recommendation based on this data."""
    response = basic_chain.invoke(prompt)
    return response.content

# ====== Define the Agents that perform tasks in the Crew ======

# Agent 1 - Researcher
# Responsible for querying the video_games collection and gathering raw info
researcher = Agent(
    role="Game Research Specialist",
    goal="Find relevant game information from the knowledge base based on the user's request",
    backstory="""You are a gaming expert with deep knowledge of video games.
    Use the provided tool to find data relevant to the user's query. 
    Return the exact data that the tool provides, without adding to it.""",
    tools=[search_game_knowledge],  # Give access to the VectorDB search tool
    max_iter=1,  # force stop after 1 iteration (no endless looping)
    llm=llm,  # Point at local Ollama instead of looking for OpenAI keys by default
    verbose=True  # shows the agent's reasoning
)

# Agent 2 - Recommender
# Takes the researcher's findings and writes a helpful, friendly response
recommender = Agent(
    role="Game Recommendation Specialist",
    goal="Take researched game data and "
         "generate a personalized string-based recommendation for the user",
    backstory="""You are a friendly game store assistant who specializes in 
    matching players with games they will love. You take raw research and 
    turn it into clear, enthusiastic recommendations ONLY in string format. no JSON.""",
    tools = [process_game_data], # Give access to the data processing tool
    llm=llm,
    max_iter=1,
    verbose=True
)

# ========= Building a Crew: Defining Tasks and the Agents that will perform them =========

def build_crew(user_query: str) -> Crew:

    research_task = Task(
        description=f"""Search the game knowledge base for information relevant to this request: 
        '{user_query}'. 
        Return all relevant game titles, genres, and details you find.""",
        expected_output="A list of relevant games with their details from the VectorDB",
        agent=researcher
    )

    recommendation_task = Task(
        description=f"""You have received game search results from the researcher. 
        Pass ONLY those exact search results as the game_data argument to the process_game_data tool.
        Do not invent or add any games not present in the researcher's output.
        The user asked: '{user_query}'""",
        expected_output="A friendly recommendation based only on the researched games",
        agent=recommender,
        context=[research_task]
    )

    return Crew(
        agents=[researcher, recommender],
        tasks=[research_task, recommendation_task],
        verbose=True,
    )