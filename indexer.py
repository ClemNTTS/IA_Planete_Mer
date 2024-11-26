from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import os
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Chemin vers le dossier contenant les fichiers PDF
folder_path = "data"
# Liste pour stocker les documents chargés
documents = []

def is_page_empty(text: str) -> bool:
    """
    Vérifie si une page est vide ou contient peu de texte exploitable.
    
    Args:
        text (str): Le texte à vérifier
    Returns:
        bool: True si la page est considérée comme vide
    """
    return not text.strip() or len(text.strip()) < 10

def extract_text_with_ocr(pdf_path: str, page_num: int = None) -> str:
    """
    Extrait le texte d'un PDF avec OCR en utilisant Tesseract.
    
    Args:
        pdf_path (str): Chemin vers le fichier PDF à traiter
        page_num (int, optional): Numéro de page spécifique à traiter
        
    Returns:
        str: Le texte extrait du PDF
    """
    try:
        # Conversion du PDF en images
        images = convert_from_path(pdf_path)
        
        # Si un numéro de page est spécifié et valide
        if page_num is not None and 0 <= page_num < len(images):
            return pytesseract.image_to_string(images[page_num], lang='fra')
        
        # Sinon, traiter toutes les pages
        extracted_text = "\n".join(
            pytesseract.image_to_string(img, lang='fra')
            for img in images
        )
        
        return extracted_text
        
    except Exception as e:
        logging.error(f"Erreur OCR pour {pdf_path}: {str(e)}")
        return ""

# Parcourir tous les fichiers du répertoire
total_files = len([f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')])
processed_files = 0

for filename in os.listdir(folder_path):
    if not filename.lower().endswith('.pdf'):
        continue
        
    file_path = os.path.join(folder_path, filename)
    processed_files += 1
    logging.info(f"Traitement du fichier {processed_files}/{total_files}: {filename}")
    
    try:
        # Ouvrir le fichier PDF avec PyPDF2
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Extraire le texte de chaque page du PDF
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                
                # Si la page est vide ou illisible, utiliser l'OCR
                if is_page_empty(page_text):
                    logging.info(f"Utilisation de l'OCR pour {filename}, page {page_num + 1}")
                    page_text = extract_text_with_ocr(file_path, page_num)
                
                text += page_text + "\n"
            
            # Créer un objet Document avec le texte extrait et des métadonnées enrichies
            document = Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "filename": filename,
                    "pages": len(reader.pages),
                    "size": os.path.getsize(file_path)
                }
            )
            documents.append(document)
            
    except Exception as e:
        logging.error(f"Erreur lors du traitement de {filename}: {str(e)}")
        continue

# Afficher le nombre de documents chargés
logging.info(f"Nombre de documents PDF chargés : {len(documents)}")

# Étape 2 : Diviser les documents en chunks avec des paramètres optimisés
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", " ", ""] # Séparateurs plus intelligents
)

texts = text_splitter.split_documents(documents)
logging.info(f"Nombre de chunks générés : {len(texts)}")

# Étape 3 : Initialiser le modèle Hugging Face pour les embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}  # Explicite l'utilisation du CPU
)

# Étape 4 : Créer et persister la base de données Chroma
persist_directory = "./db-planet-mer"

try:
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    logging.info("Vectorstore créé avec succès")
    vectorstore.persist()
    logging.info("Vectorstore persisté avec succès")
    
except Exception as e:
    logging.error(f"Erreur lors de la création du vectorstore: {str(e)}")
    raise

# Étape 5 : Charger et vérifier la base de données
try:
    loaded_vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    doc_count = len(loaded_vectorstore.get())
    logging.info(f"Nombre de documents dans le vectorstore chargé : {doc_count}")
    
    if doc_count != len(texts):
        logging.warning(
            f"Attention: Différence entre le nombre de chunks ({len(texts)}) "
            f"et le nombre de documents dans le vectorstore ({doc_count})"
        )
        
except Exception as e:
    logging.error(f"Erreur lors de la vérification du vectorstore: {str(e)}")
    raise