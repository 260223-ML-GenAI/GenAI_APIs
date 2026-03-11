import streamlit as st

from services.langchain_service import get_basic_chain

# Importing our basic chain
basic_chain = get_basic_chain()

st.title("*sigh*... GamerStopGPT")

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

    # Invoke basic chain and save the response
    response = basic_chain.invoke({"input": prompt})
    response_text = response.content  # Grabbing just the content, not the full object

    # Display LLM response
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.write(response_text)