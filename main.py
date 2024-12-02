import streamlit as st
from streamlit_chat import message
from typing import Set
from ingestion import PDFIngestion
from generate.chatbot import get_response


# Check if the vector store is already created in session state
if 'vector_store' not in st.session_state:
    # Create the vector store only once, when the app first loads
    pdf_path = 'CONTRATO_AP000000718.pdf'
    ingest = PDFIngestion(pdf_path, index='chatqa')
    st.session_state['vector_store'] = ingest.create_vector_store()
vector_store = st.session_state['vector_store']

st.set_page_config(page_title="QA ChatBot", layout="wide")
st.header("QA CHATBOT")

# Initialize chat history and answer history in session state if not present
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""


prompt = st.text_input(
    "Prompt", 
    placeholder="Enter your prompt here...", 
    key="prompt_input",
    value=st.session_state["input"]
)


# When the user submits a prompt, generate the response
if prompt:
    with st.spinner("Generating response..."):
        out = get_response(prompt, chat_history=st.session_state['chat_history'], vector_store=vector_store)
        final_response = f"{out['result']} \n\n\n"
        
        # Update chat history and store user and AI responses
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(final_response)
        st.session_state['chat_history'].append(("human", prompt))
        st.session_state['chat_history'].append(("ai", out['result']))
        
        # if out.get('source_pages'):
        #     st.sidebar.write("Source Pages:", out['source_pages'])

# Display the chat history
if st.session_state["chat_answers_history"]:
    for ques, ans in zip(reversed(st.session_state["user_prompt_history"]), reversed(st.session_state["chat_answers_history"])):
        message(ques, is_user=True)
        message(ans)

# st.markdown("""
# <style>
# .css-1d391kg p {
#     word-wrap: break-word;
# }
# </style>
# """, unsafe_allow_html=True)
