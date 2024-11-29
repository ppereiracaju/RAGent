import time
from vector_db import VectorStore
from utils import call_claude_rag, call_tavily_web_search, assess_confidence, synthesize_information
import logging

def main(user_query, initialize=False, pdf_path=None):
    # Initialize vector store
    vector_store = VectorStore(persist_directory="./chroma_db")
    
    if initialize and pdf_path:
        vector_store.initialize_from_pdf(pdf_path)
    elif not vector_store.db:
        logging.error("No vector store found. Please initialize with a PDF first.")
        return "Error: Vector store not initialized. Please provide a PDF file first."
    
    # Step 1: Initial Query Processing
    internal_response = vector_store.search(user_query)
    
    # Step 2: Reasoning and Uncertainty Detection
    rag_response = call_claude_rag(user_query, internal_response)
    time.sleep(1)
    
    if isinstance(rag_response, list):
        rag_response = " ".join(str(item) for item in rag_response)  # Convert each TextBlock to string
    
    if assess_confidence(rag_response):
        return rag_response
    
    # Step 3: Web Search and Validation
    web_results = call_tavily_web_search(user_query)
    time.sleep(1)
    synthesized_response = synthesize_information(rag_response, web_results)
    
    return synthesized_response

# Example usage
if __name__ == "__main__":
    pdf_path = "france.pdf"
    
    # Initialize the vector DB (only needed once)
    user_query = "What is the population of France in november 202?"
    response = main(user_query, initialize=False, pdf_path=pdf_path)
    print(response)