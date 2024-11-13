import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Initialisation des embeddings et de la base de données Chroma
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory="./db-planet-mer", embedding_function=embeddings)
retriever = db.as_retriever(
    search_type="similarity",
    mmr=True,
    hybrid=True,
    search_kwargs = {"k": 5}
)

# Configuration du modèle de langage
llm = ChatOllama(model="llama3.2", keep_alive="3h", max_tokens=512, temperature=0)

# Création du modèle de prompt
template = """<bos><start_of_turn>user
Answer the question based only on the following context and provide a detailed, accurate response. Please write in full sentences with proper spelling and punctuation. If the context allows, use lists for clarity. 
If the answer is not found within the context, kindly respond that you are unable to provide an answer. 
You are an expert fisherman with a deep understanding of marine biology and paleontology. 
Feel free to sprinkle in some humor and intriguing tidbits related to the world of fishing and ancient creatures!

CONTEXT: {context}

QUESTION: {question}

<end_of_turn>
<start_of_turn>model
ANSWER:"""
prompt = ChatPromptTemplate.from_template(template)

# Chaine RAG (Retrieve-then-Answer Generation)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Fonction pour générer une réponse à partir de la chaîne RAG
def generate_response(input_text):
    answer = ""
    for chunk in rag_chain.stream(input_text):
        answer += chunk.content
    return answer

# Interface utilisateur avec Streamlit
st.title("ChatBot Planète Mer")
st.write("Planète Mer est une association pour les pêcheurs")

# Formulaire d'entrée utilisateur
with st.form("llm-form"):
    text = st.text_area("Entrez votre message...")
    submit = st.form_submit_button("Demander à l'IA")

# Initialisation de l'historique de chat
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

# Gestion de la soumission du formulaire
if submit and text:
    with st.spinner("Réponse en cours..."):
        response = generate_response(text)
        st.session_state['chat_history'].append({"user": text, "ollama": response})
        st.write(response)

# Affichage de l'historique de chat
st.write("## Chat History")
for chat in reversed(st.session_state['chat_history']):
    st.write(f"**😎 Vous**: {chat['user']}")
    st.write(f"**🧠 Assistant**: {chat['ollama']}")
    st.write("---")

# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_community.chat_models import ChatOllama
# import streamlit as st
# #from langchain_ollama import ChatOllama

# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser

# embeddings = OllamaEmbeddings(model="nomic-embed-text")

# db = Chroma(persist_directory="./db-planet-mer", embedding_function=embeddings)

# retriever = db.as_retriever(
#     search_type="similarity",
#     mmr=True,
#     hybrid=True,
#     search_kwargs = {"k": 5}
# )

# my_llm = "llama3.2"

# llm = ChatOllama(model=my_llm,
#                  keep_alive="3h",
#                  max_tokens=512,
#                  temperature=0)

# # Create prompt template
# template = """<bos><start_of_turn>user
# Answer the question based only on the following context and provide a detailed, accurate response. Please write in full sentences with proper spelling and punctuation. If the context allows, use lists for clarity. 
# If the answer is not found within the context, kindly respond that you are unable to provide an answer. 
# You are an expert fisherman with a deep understanding of marine biology and paleontology. 
# Feel free to sprinkle in some humor and intriguing tidbits related to the world of fishing and ancient creatures!

# CONTEXT: {context}

# QUESTION: {question}

# <end_of_turn>
# <start_of_turn>model
# ANSWER:"""
# prompt = ChatPromptTemplate.from_template(template)

# # Create the RAG chain using LCEL with prompt printing and streaming output
# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
# )

# # Function to ask questions
# def ask_question(question):
#     print("Answer:\n\n", end=" ", flush=True)
#     for chunk in rag_chain.stream(question):
#         print(chunk.content, end="", flush=True)
#     print("\n")

# # Example usage
# if __name__ == "__main__":
#     while True:
#         user_question = input("Ask a question (or type 'quit' to exit): ")
#         if user_question.lower() == 'quit':
#             break
#         answer = ask_question(user_question)
#         # print("\nFull answer received.\n")