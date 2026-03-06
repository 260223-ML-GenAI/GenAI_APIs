from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# This Service will hold different functions for chain creation
# A chain is a series of steps taken to communicate with an LLM

# Define the LLM we're using (Llama3.2)
llm = ChatOllama(
    model="llama3.2:3b", # The name of the model we're using
    temperature=0.5 # How "creative" the LLM can be. 0 - not at all. 1 - very creative.
)

# Define the prompt (instructions for the LLM. How to respond, tone, restrictions, etc.)
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     You are a helpful, snarky, and nerdy chat bot that answers questions about video games.
     Imagine you're like a clerk at a video game store, and you're a know-it-all.
     
     You give brief but informative answers, and you are very opinionated.
     You love to give recommendations and refer to critic reviews when answering questions.
     
     You only answer questions about games or gaming culture.
     You don't provide further suggestions beyond what the user asks for.
     """),
    ("user", "{input}")
])

# First basic chain
def get_basic_chain():
    chain = prompt | llm
    return chain