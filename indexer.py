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

# Debug logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    A class to process and index PDF documents into a Milvus vector database.
    Handles text extraction, cleaning, and vectorization of document content.
    """
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        chunk_size: int = 500,
        chunk_overlap: int = 200,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "planet_mer"
    ):
        """
        Initialize the DocumentProcessor with database and processing parameters.

        Args:
            host: Milvus host address
            port: Milvus port number
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
            model_name: Name of the embedding model to use
            collection_name: Name of the Milvus collection
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.collection_name = collection_name
        self.host = host
        self.port = port
        
        # Initialize database connection and embedding model
        self._init_milvus_connection()
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)
        
        # Initialize text splitter with custom separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\nArticle", "\n\nCHAPTER", "\n\nTITLE", "\n\n", "\n", ". ", " ", ""]
        )

    def _init_milvus_connection(self) -> None:
        """Initialize connection to Milvus database and verify status."""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            status = connections.get_connection_addr("default")
            collections = utility.list_collections()
            logger.info(f"Successfully connected to Milvus. Status: {status}")
            logger.info(f"Existing collections: {collections}")
        except Exception as e:
            logger.error(f"Milvus connection error: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        text = text.replace("_", " ").replace("\xa0", " ").replace("'", "'")
        text = " ".join(text.split())
        # Normalize multiple newlines to double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'Article\s+(\d+)\s*:', r'Article \1 :', text)
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable())
        return text

    def split_by_article(self, text: str) -> List[str]:
        """
        Split text into chunks based on article boundaries.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        article_pattern = r'(Article \d+\s*:.*?)(?=Article \d+\s*:|$)'
        articles = re.finditer(article_pattern, text, re.DOTALL)
        
        for article in articles:
            chunk = article.group(1).strip()
            if chunk and len(chunk.split()) > 20:
                chunks.append(chunk)
        
        # Fallback to size-based splitting if no articles found
        if not chunks:
            chunks = self.text_splitter.split_text(text)

        logger.info(f"Number of chunks created: {len(chunks)}")
        return chunks

    def extract_text_with_pypdf2(self, pdf_path: str) -> str:
        """
        Extract text from PDF using PyPDF2.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return "\n".join(page.extract_text() for page in pdf_reader.pages)
        except Exception as e:
            logger.error(f"PyPDF2 error for {pdf_path}: {str(e)}")
            return ""

    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """
        Extract text from PDF using OCR when regular extraction fails.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            images = convert_from_path(pdf_path)
            return "\n".join(pytesseract.image_to_string(img, lang='eng') for img in images)
        except Exception as e:
            logger.error(f"OCR error for {pdf_path}: {str(e)}")
            return ""

    def process_single_pdf(self, pdf_path: str) -> List[Document]:
        """
        Process a single PDF file and convert it to document chunks.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of processed Document objects
        """
        try:
            logger.info(f"Processing {pdf_path}")
            
            # Try regular PDF extraction first
            text = self.extract_text_with_pypdf2(pdf_path)
            extraction_method = "text"
            
            # Fall back to OCR if needed
            if not text.strip():
                logger.info(f"Attempting OCR for {pdf_path}")
                text = self.extract_text_with_ocr(pdf_path)
                extraction_method = "ocr"
            
            if not text.strip():
                logger.error(f"No text extracted from {pdf_path}")
                return []
            
            text = self.clean_text(text)
            logger.info(f"Text length after cleaning: {len(text)}")
            
            chunks = self.split_by_article(text)
            
            documents = []
            for i, chunk in enumerate(chunks):
                if len(chunk.split()) >= 20:  # Minimum chunk size threshold
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
            
            logger.info(f"Number of documents created for {pdf_path}: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def process_directory(self, data_dir: str, max_workers: int = 4) -> List[Document]:
        """
        Process all PDF files in a directory using parallel execution.
        
        Args:
            data_dir: Directory containing PDF files
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of processed Document objects
        """
        if not os.path.exists(data_dir):
            logger.error(f"Directory {data_dir} does not exist")
            return []
            
        pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
        pdf_files = [os.path.join(data_dir, f) for f in pdf_files]
        
        if not pdf_files:
            logger.error(f"No PDF files found in {data_dir}")
            return []
            
        logger.info(f"Number of PDF files found: {len(pdf_files)}")
        
        documents = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(self.process_single_pdf, pdf_files),
                total=len(pdf_files),
                desc="Processing PDFs"
            ))
            documents = [doc for result in results if result for doc in result]
                
        logger.info(f"Total number of documents processed: {len(documents)}")
        return documents

    def create_vectorstore(self, documents: List[Document]) -> None:
        """
        Create a vector store from processed documents.
        
        Args:
            documents: List of Document objects to index
        """
        try:
            if not documents:
                raise ValueError("No documents to index")
            
            # Verify chunk sizes
            for doc in documents:
                if len(doc.page_content.split()) < 20:
                    logger.warning(f"Potentially too short chunk detected: {doc.page_content}")
            
            # Clean existing collection
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Collection {self.collection_name} dropped")
            
            # Create vector store
            vectorstore = Milvus.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                collection_name=self.collection_name,
                connection_args={"host": self.host, "port": self.port}
            )
            
            # Verify indexing
            if utility.has_collection(self.collection_name):
                try:
                    test_results = vectorstore.similarity_search("test", k=1)
                    logger.info("Search test successful")
                    logger.info(f"Vector store created successfully - {len(documents)} documents indexed")
                except Exception as search_error:
                    logger.error(f"Search test error: {str(search_error)}")
                    raise
            else:
                raise Exception("Collection was not created properly")
                
        except Exception as e:
            logger.error(f"Vector store creation error: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def process_and_index(self, data_dir: str, max_workers: int = 4) -> None:
        """
        Complete pipeline for processing and indexing documents.
        
        Args:
            data_dir: Directory containing PDF files
            max_workers: Maximum number of parallel workers
        """
        try:
            logger.info(f"Starting directory processing: {data_dir}")
            documents = self.process_directory(data_dir, max_workers)
            
            if documents:
                logger.info("Starting document indexing")
                self.create_vectorstore(documents)
                logger.info("Indexing completed successfully")
            else:
                logger.error("No documents were processed successfully")
                
        except Exception as e:
            logger.error(f"Processing and indexing error: {str(e)}")
            logger.error(traceback.format_exc())
            raise

if __name__ == "__main__":
    try:
        processor = DocumentProcessor()
        data_directory = "data"
        
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
            logger.info(f"Created directory {data_directory}")
        
        processor.process_and_index(data_directory)
        
    except Exception as e:
        logger.error(f"Main error: {str(e)}")
        logger.error(traceback.format_exc())