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

    # This is a basic chain - it consists of:
        # a prompt
        # an LLM that will use that prompt to answer user queries
    # This pipe-based syntax is LCEL (LangChain Expression Language)
    chain = prompt | llm

    # This chain will get invoked by endpoints in our router!
    return chain

# Sequential Chain - take the initial LLM output and feed it into another LLM invocation
def get_sequential_chain():

    """ This chain is for customer support.
     The first LLM response is our typical sarcatic LLM tone
     We want customer support chats to take a more professional and kind tone """

    # First LLM, just get a basic chain from above
    first_chain = get_basic_chain()

    # Define a new prompt for the second LLM call
    second_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and kind customer support chatbot. "
                   "You respond to hypothetical customer support chats. "
                   "Take the initial response from the first LLM and rewrite it. "
                   "Change sarcastic tone of the first response into a more kind response. "
                   "ONLY return the rewritten response, imagine this is a chat to a real customer. "
                   "Don't include any preamble text, ONLY the response to the user. "
                   "You don't need to comfort the user, just tell them a solution. "),
        ("user", "Initial LLM response: {input}")
    ])

    # Second chain, same model, different prompt
    second_chain = second_prompt | llm

    # Finally, build a sequential chain (take first LLM output and feed it to the second LLM)
    final_chain = first_chain | second_chain
    # (prompt | llm | second_prompt | llm)

    return final_chain
