import os
import logging
import traceback
import pytesseract
from pdf2image import convert_from_path
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from pymilvus import connections, utility
from typing import List
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import PyPDF2
import re

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        chunk_size: int = 500,
        chunk_overlap: int = 200,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "planet_mer"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.collection_name = collection_name
        self.host = host
        self.port = port
        
        self._init_milvus_connection()  # Connexion à Milvus
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)
        
        # Initialisation du découpeur de texte pour créer des chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\nArticle", "\n\nCHAPITRE", "\n\nTITRE", "\n\n", "\n", ". ", " ", ""]
        )

    def clean_text(self, text: str) -> str:
        """Nettoie et normalise le texte"""
        text = text.replace("_", " ").replace("\xa0", " ").replace("'", "'")
        text = " ".join(text.split())
        text = re.sub(r'\n{3,}', '\n\n', text)  # Remplace les multiples sauts de ligne par deux
        text = re.sub(r'Article\s+(\d+)\s*:', r'Article \1 :', text)
        text = ''.join(char for char in text if char.isprintable())  # Supprimer les caractères non imprimables
        return text

    def split_by_article(self, text: str) -> List[str]:
        """Découpe le texte en préservant la structure des articles"""
        chunks = []
        article_pattern = r'(Article \d+\s*:.*?)(?=Article \d+\s*:|$)'
        articles = re.finditer(article_pattern, text, re.DOTALL)
        
        for article in articles:
            chunk = article.group(1).strip()
            if chunk and len(chunk.split()) > 20:
                chunks.append(chunk)
        
        if not chunks:
            chunks = self.text_splitter.split_text(text)  # Si aucun article trouvé, découpe par taille

        logger.info(f"Nombre de chunks créés: {len(chunks)}")
        return chunks

    def extract_text_with_pypdf2(self, pdf_path: str) -> str:
        """Extrait le texte d'un PDF avec PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return "\n".join(page.extract_text() for page in pdf_reader.pages)
        except Exception as e:
            logger.error(f"Erreur PyPDF2 pour {pdf_path}: {str(e)}")
            return ""

    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extrait le texte d'un PDF avec OCR"""
        try:
            images = convert_from_path(pdf_path)
            return "\n".join(pytesseract.image_to_string(img, lang='fra') for img in images)
        except Exception as e:
            logger.error(f"Erreur OCR pour {pdf_path}: {str(e)}")
            return ""

    def process_single_pdf(self, pdf_path: str) -> List[Document]:
        """Traite un seul fichier PDF"""
        try:
            logger.info(f"Traitement de {pdf_path}")
            
            text = self.extract_text_with_pypdf2(pdf_path)
            extraction_method = "text"
            
            # Si PyPDF2 ne trouve pas de texte, utilise OCR
            if not text.strip():
                logger.info(f"Tentative OCR pour {pdf_path}")
                text = self.extract_text_with_ocr(pdf_path)
                extraction_method = "ocr"
            
            if not text.strip():  # Si le texte est toujours vide, on ne continue pas
                logger.error(f"Aucun texte extrait de {pdf_path}")
                return []
            
            text = self.clean_text(text)  # Nettoyage du texte extrait
            logger.info(f"Longueur du texte après nettoyage: {len(text)}")
            
            # Découpe le texte en chunks (articles)
            chunks = self.split_by_article(text)
            
            documents = []
            for i, chunk in enumerate(chunks):
                if len(chunk.split()) >= 20:  # Si le chunk est suffisamment long
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": pdf_path,
                            "extraction_method": extraction_method,
                            "filename": os.path.basename(pdf_path),
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Nombre de documents créés pour {pdf_path}: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {pdf_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def process_directory(self, data_dir: str, max_workers: int = 4) -> List[Document]:
        """Traite tous les PDF d'un répertoire en parallèle"""
        if not os.path.exists(data_dir):
            logger.error(f"Le répertoire {data_dir} n'existe pas")
            return []
            
        pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
        pdf_files = [os.path.join(data_dir, f) for f in pdf_files]
        
        if not pdf_files:
            logger.error(f"Aucun fichier PDF trouvé dans {data_dir}")
            return []
            
        logger.info(f"Nombre de fichiers PDF trouvés: {len(pdf_files)}")
        
        documents = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(self.process_single_pdf, pdf_files),
                total=len(pdf_files),
                desc="Traitement des PDFs"
            ))
            documents = [doc for result in results if result for doc in result]
                
        logger.info(f"Nombre total de documents traités: {len(documents)}")
        return documents

    def create_vectorstore(self, documents: List[Document]) -> None:
        """Crée le vectorstore à partir des documents"""
        try:
            if not documents:
                raise ValueError("Aucun document à indexer")
            
            # Vérification des chunks
            for doc in documents:
                if len(doc.page_content.split()) < 20:
                    logger.warning(f"Chunk potentiellement trop court détecté: {doc.page_content}")
            
            # Nettoyage de la collection existante
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Collection {self.collection_name} supprimée")
            
            # Création du vectorstore
            vectorstore = Milvus.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                collection_name=self.collection_name,
                connection_args={"host": self.host, "port": self.port}
            )
            
            # Vérification de l'indexation
            if utility.has_collection(self.collection_name):
                # Test de recherche pour vérifier que l'indexation fonctionne
                try:
                    test_results = vectorstore.similarity_search("test", k=1)
                    logger.info("Test de recherche réussi")
                    logger.info(f"Vectorstore créé avec succès - {len(documents)} documents indexés")
                except Exception as search_error:
                    logger.error(f"Erreur lors du test de recherche: {str(search_error)}")
                    raise
            else:
                raise Exception("La collection n'a pas été créée correctement")
                
        except Exception as e:
            logger.error(f"Erreur lors de la création du vectorstore: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def process_and_index(self, data_dir: str, max_workers: int = 4) -> None:
        """Pipeline complet de traitement et d'indexation"""
        try:
            logger.info(f"Début du traitement du répertoire: {data_dir}")
            documents = self.process_directory(data_dir, max_workers)
            
            if documents:
                logger.info("Début de l'indexation des documents")
                self.create_vectorstore(documents)
                logger.info("Indexation terminée avec succès")
            else:
                logger.error("Aucun document n'a pu être traité correctement")
                
        except Exception as e:
            logger.error(f"Erreur lors du traitement et de l'indexation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

if __name__ == "__main__":
    try:
        processor = DocumentProcessor()
        data_directory = "data"
        
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
            logger.info(f"Répertoire {data_directory} créé")
        
        processor.process_and_index(data_directory)
        
    except Exception as e:
        logger.error(f"Erreur principale: {str(e)}")
        logger.error(traceback.format_exc())
