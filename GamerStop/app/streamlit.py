import streamlit as st

from services.langchain_service import get_basic_chain

# Importing our basic chain
basic_chain = get_basic_chain()

# Basic header
st.title("*sigh*... GamerStopGPT")

# Sidebar with one (1) working button
with st.sidebar:
    st.header("About")
    st.write("This is a Streamlit demo full of generic and easy to use components."
             "Use Streamlit to make quick and easy frontends to interact with the API!")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history. This isn't real memory, just for rendering in the UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
# Save (write) new messages to session_state.messages for display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about games..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Invoke basic chain and save the response (WITH A LOADING ANIMATION!)
    with st.spinner("Thinking..."):
        response = basic_chain.invoke({"input": prompt})

    # Grabbing just the content for display, not the full object
    response_text = response.content

    # Display LLM response
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.write(response_text)