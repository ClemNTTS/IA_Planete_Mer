import streamlit as st
from langchain_ollama import ChatOllama

st.title("ChatBot planete mer")

st.write("Planete mer est une association pour les pécheurs")

with st.form("llm-form"): 
    text = st.text_area("Entrez votre message...")
    submit = st.form_submit_button("Demander à l'ia")

def generate_response(input_text):
    model = ChatOllama(
        model="llama3.2",
        base_url="http://localhost:11434"
    )
    response = model.invoke(input_text)
    return response.content

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

if submit and text:
    with st.spinner("Réponse en cours..."):
          response = generate_response(text)
          st.session_state['chat_history'].append({"user": text, "ollama": response})
          st.write(response)

st.write("## Chat History")
for chat in reversed(st.session_state['chat_history']):
    st.write(f"**😎 Vous**: {chat['user']}")
    st.write(f"**🧠 Assistant**: {chat['ollama']}")
    st.write("---")

