from io import StringIO

import streamlit as st

st.header("Summarize Text Demo")

uploaded_file = st.file_uploader("Choose a TXT file:")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Popup with the uploaded TXT content
    st.warning("Here's the content of your uploaded file:")
    st.code(string_data)