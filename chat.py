import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from pymilvus import connections, utility

# Connexion Milvus
connections.connect(
   alias="default",
   host="localhost", 
   port="19530"
)

# Configuration
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
collection_name = "planet_mer"
db = Milvus(
   embedding_function=embedding_model,
   collection_name=collection_name,
   connection_args={"host": "localhost", "port": "19530"}
)

retriever = db.as_retriever(
   search_type="similarity",
   search_kwargs={"k": 2}
)

# LLM Config
llm = ChatOllama(model="llama3.2", keep_alive="3h", max_tokens=512, temperature=0.1)

# Tests de vérification
print("Collection existe:", utility.has_collection("planet_mer"))
print("Test requête:", len(retriever.get_relevant_documents("test")))
print("Nombre documents:", db.col.num_entities)

def format_docs(docs):
   formatted_docs = [doc.page_content for doc in docs]
   sources = [doc.metadata.get('source', 'Source inconnue') for doc in docs]
   return "\n\n".join(formatted_docs), sources

def generate_response_with_sources(retriever, question):
   docs = retriever.get_relevant_documents(question)
   context, sources = format_docs(docs)
   
   messages = [
    {"role": "system", "content": "Vous êtes un expert en biologie marine et en pêche. Vous devez répondre de manière précise sur les espèces marines réglementées en Méditerranée française."},
    {"role": "user", "content": "Pouvez-vous me donner une liste des espèces marines réglementées en Méditerranée, y compris la dorade rose, le thon germon, et d'autres espèces mentionnées dans les règlements halieutiques ?"},
       {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
   ]
   
   chain_response = llm.invoke(messages)
   return chain_response.content, sources

# Streamlit UI
st.set_page_config(page_title="Planète Mer ChatBot", page_icon="🐠")

st.markdown("""
   <style>
       .user-message {
           background-color: #e3f2fd;
           color: #0d47a1;
           padding: 10px;
           border-radius: 10px;
           margin: 5px 0;
           max-width: 80%;
           align-self: flex-end;
       }
       .assistant-message {
           background-color: #f1f8e9;
           color: #33691e;
           padding: 10px;
           border-radius: 10px;
           margin: 5px 0;
           max-width: 80%;
           align-self: flex-start;
       }
   </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
   st.session_state.messages = [
       {"role": "assistant", "content": ("Bonjour! Je suis là pour répondre à vos questions sur la pêche et la vie marine. Comment puis-je vous aider? 🎣", [])}
   ]

with st.sidebar:
   st.title("ChatBot Planète Mer")
   if st.button("Effacer l'historique"):
       st.session_state.messages = [
           {"role": "assistant", "content": ("Bonjour! Je suis là pour répondre à vos questions sur la pêche et la vie marine. Comment puis-je vous aider? 🎣", [])}
       ]
   
   st.markdown("### Historique des conversations")
   for message in st.session_state.messages[1:]:
       if message["role"] == "user":
           st.markdown(f"**😎 Vous:** {message['content'][:100]}...")
       else:
           response, _ = message["content"]
           st.markdown(f"**🤖 Assistant:** {response[:100]}...")

st.title("Bonjour !")

for message in st.session_state.messages:
   if message["role"] == "user":
       st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
   else:
       response, sources = message["content"]
       st.markdown(f'<div class="assistant-message">{response}</div>', unsafe_allow_html=True)
       if sources:
           with st.expander("Sources"):
               for source in sources:
                   st.markdown(f"- {source}")

if prompt := st.chat_input("Posez votre question ici..."):
   st.session_state.messages.append({"role": "user", "content": prompt})
   st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
   
   with st.spinner("Réflexion en cours..."):
       response, sources = generate_response_with_sources(retriever, prompt)
       st.session_state.messages.append({"role": "assistant", "content": (response, sources)})
       st.markdown(f'<div class="assistant-message">{response}</div>', unsafe_allow_html=True)
       if sources:
           with st.expander("Sources"):
               for source in sources:
                   st.markdown(f"- {source}")