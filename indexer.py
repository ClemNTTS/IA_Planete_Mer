from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Load documents from a directory
loader = DirectoryLoader("data")

print("dir loaded loader")

documents = loader.load()

print(len(documents))

# Create embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

# Create Recursive Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
  #  separators=["\n\n", "\n", ".", "!", "?", ",", " "]
)

# Split documents into chunks
texts = text_splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=texts, 
    embedding=embeddings,
    persist_directory="./db-planet-mer")

print("vectorstore created")

# Verify the persistence of the data
loaded_vectorstore = Chroma(persist_directory="./db-planet-mer", embedding_function=embeddings)
print(f"Number of documents in loaded vectorstore: {len(loaded_vectorstore)}")
