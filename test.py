from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document  # Import nécessaire pour créer des objets Document

import os
import PyPDF2

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

loaded_vectorstore = Chroma(
    persist_directory="./db-planet-mer",
    embedding_function=embedding_model
)

print(loaded_vectorstore.get())

print(f"Nombre de documents dans le vectorstore chargé : {len(loaded_vectorstore.get())}")
