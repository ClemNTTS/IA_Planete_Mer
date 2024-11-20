from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Étape 1 : Charger les documents depuis un dossier
loader = DirectoryLoader("data")  # Chemin vers vos fichiers
print("Documents chargés depuis le répertoire.")
documents = loader.load()
print(f"Nombre de documents chargés : {len(documents)}")

# Étape 2 : Diviser les documents en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False
)
texts = text_splitter.split_documents(documents)
print(f"Nombre de chunks générés : {len(texts)}")

# Étape 3 : Initialiser le modèle Hugging Face pour les embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Étape 4 : Créer et persister la base de données Chroma
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embedding_model,
    persist_directory="./db-planet-mer"  # Chemin où persister les embeddings
)
print("Vectorstore créé et persisté.")

# Étape 5 : Charger la base de données pour vérification
loaded_vectorstore = Chroma(
    persist_directory="./db-planet-mer",
    embedding_function=embedding_model
)
print(f"Nombre de documents dans le vectorstore chargé : {len(loaded_vectorstore.get())}")
