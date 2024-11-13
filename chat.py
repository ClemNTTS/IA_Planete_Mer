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
    search_type="mmr",

    search_kwargs={ 
        "k": 5,

    },
#     retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs={ 
#         "k": 5,
#         "fetch_k": 20,                 # Récupère plus de candidats initiaux
#         "lambda_mult": 0.7,            # Paramètre MMR pour l'équilibre pertinence/diversité
#         "filter": {"metadata_field": "value"} # Filtre sur les métadonnées
#     },
#     hybrid_search_kwargs={  
#         "alpha": 0.5,                  # Poids entre recherche vectorielle et par mots-clés  
#         "boost_mode": "multiply"       # Mode de combinaison des scores
#     }
# )

)
# Configuration du modèle de langage
llm = ChatOllama(model="llama3.2", keep_alive="3h", max_tokens=512, temperature=5)

# Template du prompt
template = """<bos><start_of_turn>user
Answer the question based only on the following context and provide a detailed, accurate response. Please write in full sentences with proper spelling and punctuation. If the context allows, use lists for clarity. 
If the answer is not found within the context, kindly respond that you are unable to provide an answer.
You are an expert fisherman with a deep understanding of marine biology and paleontology.
Feel free to sprinkle in some humor and intriguing tidbits related to the world of fishing and ancient creatures!

Your response should follow this format:
1. Your detailed answer
2. End your response with: "**Sources:** [List the document names/titles where you found this information]"

CONTEXT: with your data, {context}

QUESTION: {question}

<end_of_turn>
<start_of_turn>model
ANSWER:"""

prompt = ChatPromptTemplate.from_template(template)

# Implémenter la memory pour améliorer la réponse

# FAISS

def format_docs(docs):
    formatted_docs = []
    doc_titles = []
    for doc in docs:
        formatted_docs.append(doc.page_content)
        if hasattr(doc.metadata, 'source'):
            doc_titles.append(doc.metadata['source'])

    retriever = "\n\n".join(formatted_docs), doc_titles
    print(retriever)
    return retriever


def generate_response_with_sources(retriever, question):
    docs = retriever.get_relevant_documents(question)
    context, sources = format_docs(docs)
    
    chain_response = llm.invoke(
        prompt.format(
            context=context,
            question=question
        )
    )
    
    response = chain_response.content
    if sources:
        response += f"\n\n**Sources:** {', '.join(sources)}"
    else:
        response += "\n\n**Sources:** Documentation interne Planète Mer"
    
    return response

# Configuration de la page Streamlit
st.set_page_config(page_title="Planète Mer ChatBot", page_icon="🐠")

# Initialisation des messages si pas déjà fait
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour! Je suis là pour répondre à vos questions sur la pêche et la vie marine. Comment puis-je vous aider? 🎣"}
    ]

# Sidebar avec historique
with st.sidebar:
    st.title("ChatBot Planète Mer")
    
    # Bouton pour effacer l'historique
    if st.button("Effacer l'historique"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Bonjour! Je suis là pour répondre à vos questions sur la pêche et la vie marine. Comment puis-je vous aider? 🎣"}
        ]
    
    # Affichage de l'historique dans la sidebar
    st.markdown("### Historique des conversations")
    
    # Container pour l'historique avec scrolling
    with st.container():
        for i, message in enumerate(st.session_state.messages[1:]):  # Skip the first welcome message
            # Créer un style condensé pour l'historique
            if message["role"] == "user":
                st.markdown("**😎 Vous:**")
                st.markdown(f"{message['content'][:100]}..." if len(message['content']) > 100 else message['content'])
            else:
                st.markdown("**🤖 Assistant:**")
                content = message['content']
                # Tronquer le contenu et retirer les sources pour l'historique
                if "**Sources:**" in content:
                    content = content.split("**Sources:**")[0]
                st.markdown(f"{content[:100]}..." if len(content) > 100 else content)
            st.markdown("---")

# Zone principale de chat
st.title("Bonjour !")

# Affichage des messages dans la zone principale
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de chat
if prompt := st.chat_input("Posez votre question ici..."):
    # Ajout du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Génération de la réponse
    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            response = generate_response_with_sources(retriever, prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)