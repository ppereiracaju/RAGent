# The king RAGent
## Trying to imitate an associative human thought process

This project implements a sophisticated RAG (Retrieval-Augmented Generation) system that mimics human-like information processing by combining local knowledge with web search capabilities.

### Features

- **PDF Knowledge Base**: Processes and stores PDF documents in a vector database for efficient retrieval
- **Intelligent Search**: Uses ChromaDB for semantic search of relevant information
- **Web Integration**: Leverages Tavily API for real-time web searches when needed
- **Confidence Assessment**: Uses Claude AI to evaluate response confidence
- **Information Synthesis**: Combines local knowledge with web search results

### Architecture

The system follows a multi-step process:
1. Initial query processing through local knowledge base
2. Confidence assessment of the initial response
3. Web search integration when needed
4. Information synthesis and validation

### Requirements

- Python 3.8+
- ChromaDB for vector storage
- Claude API access
- Tavily API access

### Installation

1. Clone the repository: 