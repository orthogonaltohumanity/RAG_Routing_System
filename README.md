# Simple PDF RAG with Ollama & LangChain

This project is a straightforward implementation of a Retrieval-Augmented Generation (RAG) system in Python. It allows you to load PDF documents from a local directory, process them, and ask questions about their content using locally running language models via Ollama and the LangChain framework.

This serves as a foundational example for building RAG systems and is intended as a portfolio project demonstrating the integration of these technologies.

## Features

*   Loads multiple PDF documents from a specified directory.
*   Splits documents into manageable chunks using `RecursiveCharacterTextSplitter`.
*   Generates text embeddings locally using Ollama (e.g., `mxbai-embed-large`).
*   Stores and retrieves document chunks using ChromaDB as the vector store.
*   Uses a locally running LLM via Ollama (e.g., `phi4:14b-q4_K_M`) for generating answers.
*   Implements `MultiQueryRetriever` to potentially improve document retrieval by generating multiple versions of the user's question.
*   Answers questions based *only* on the context provided by the loaded PDFs.
*   Includes a basic command-line interface for interaction.

## Technologies Used

*   **Python 3.x**
*   **LangChain & LangChain Community:** Core framework for building LLM applications.
*   **Ollama:** For running embedding and language models locally.
    *   Embedding Model Used: `mxbai-embed-large` (configurable)
    *   LLM Used: `phi4:14b-q4_K_M` (configurable)
*   **ChromaDB:** Vector store for embeddings.
*   **PyPDFLoader:** For loading PDF documents.

## Setup & Installation

1.  **Prerequisites:**
    *   Python 3.9 or higher installed.
    *   [Ollama](https://ollama.com/) installed and running. Make sure Ollama is accessible (e.g., `ollama run phi3` works in your terminal).

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/solilei/PDF-RAG-System.git
    cd PDF-RAG-System
    ```

3.  **Set up a Virtual Environment:**
    *   **Linux/macOS:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

4.  **Install Dependencies:**
    *   *(Ensure you have created a `requirements.txt` file first! Run `pip freeze > requirements.txt` in your activated environment after installing the necessary packages like `langchain`, `langchain-community`, `langchain-ollama`, `chromadb`, `pypdf`)*
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download Ollama Models:**
    Pull the default models used by the script (or any other models you wish to configure):
    ```bash
    ollama pull mxbai-embed-large
    ollama pull phi4:14b-q4_K_M
    ```
    *(Note: The `phi4:14b-q4_K_M` model is quite large (~8GB). You can substitute a smaller model like `phi3` or `llama3:8b` if needed, but you'll need to update the `llm_model` default in the `PdfRagSystem` class.)*

## Configuration

The core configuration (Ollama model names, chunk size, chunk overlap) can be adjusted directly within the `PdfRagSystem` class `__init__` method in `rag_system.py` (or wherever you place the class definition).

```python
# Example inside PdfRagSystem __init__
def __init__(
        self,
        embed_model: str = "mxbai-embed-large", # Change embedding model here
        llm_model: str = "phi4:14b-q4_K_M",    # Change LLM here
        chunk_size: int = 1200,                # Adjust chunk size
        chunk_overlap: int = 300,              # Adjust chunk overlap
):
    # ... rest of the init method