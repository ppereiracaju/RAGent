# Set environment variable before imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2", persist_directory="./chroma_db"):
        """Initialize the vector store with the specified embedding model."""
        logging.debug(f"Initializing VectorStore with model: {model_name}")
        self.embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        self.persist_directory = persist_directory
        
        # Try to load existing DB
        if os.path.exists(persist_directory):
            try:
                self.db = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embedding_function
                )
                logging.info("Loaded existing vector store from disk")
            except Exception as e:
                logging.error(f"Error loading existing vector store: {e}")
                self.db = None
        else:
            self.db = None

    def initialize_from_pdf(self, pdf_path, chunk_size=300, chunk_overlap=50):
        """Initialize vector DB with PDF content."""
        logging.debug(f"Loading PDF: {pdf_path}")
        try:
            # Load the PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            logging.debug(f"Loaded {len(documents)} pages from PDF")

            # Split documents into chunks
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )
            docs = text_splitter.split_documents(documents)
            logging.debug(f"Split into {len(docs)} chunks")

            # Create vector store
            self.db = Chroma.from_documents(
                documents=docs,
                embedding=self.embedding_function,
                persist_directory="./chroma_db"  # Persist the database to disk
            )
            logging.debug("Vector store initialized successfully")
            return True

        except Exception as e:
            logging.error(f"Error initializing vector store: {e}")
            return False

    def search(self, query, n_results=3):
        """Search the vector store for relevant documents."""
        logging.debug(f"Searching for: {query}")
        try:
            if not self.db:
                logging.error("Vector store not initialized")
                return ""

            # Perform similarity search
            docs = self.db.similarity_search(query, k=n_results)
            
            # Combine the content from all retrieved documents
            context = "\n".join(doc.page_content for doc in docs)
            logging.debug(f"Found {len(docs)} relevant documents")
            return context

        except Exception as e:
            logging.error(f"Error searching vector store: {e}")
            return ""

    def __del__(self):
        """Cleanup when the object is destroyed."""
        if self.db:
            try:
                self.db.persist()
                logging.debug("Vector store persisted to disk")
            except Exception as e:
                logging.error(f"Error persisting vector store: {e}")