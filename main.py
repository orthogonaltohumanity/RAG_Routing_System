# main.py

import argparse
import os
import sys
import logging
from rag_system import PdfRagSystem

# Configure basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def find_pdf_files(directory_path: str) -> list[str]:
    """Finds all PDF files in the specified directory."""
    pdf_files = []
    try:
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(".pdf"):
                full_path = os.path.join(directory_path, filename)
                if os.path.isfile(full_path):
                    pdf_files.append(full_path)
    except OSError as e:
        logging.error(f"Error accessing directory {directory_path}: {e}")
        raise  # Re-raise the exception after logging
    return pdf_files


def main():
    """Main function to run the PDF RAG system."""
    parser = argparse.ArgumentParser(
        description="Chat with PDF documents locally using Ollama and LangChain."
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        required=True,
        help="Path to the directory containing PDF files."
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="mxbai-embed-large",
        help="Name of the Ollama embedding model to use."
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="phi3",
        help="Name of the Ollama chat model to use."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Chunk size for splitting documents."
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=300,
        help="Chunk overlap for splitting documents."
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="multi-pdf-rag",
        help="Name for the ChromaDB collection."
    )

    args = parser.parse_args()

    # --- Validate Directory ---
    if not os.path.isdir(args.pdf_dir):
        logging.error(f"Error: PDF directory not found at '{args.pdf_dir}'")
        print(f"Error: PDF directory not found at '{args.pdf_dir}'")
        sys.exit(1)  # Exit if the directory doesn't exist

    print(f"PDF directory found: {args.pdf_dir}")
    logging.info(f"Using PDF directory: {args.pdf_dir}")

    # --- Find PDF Files ---
    try:
        pdf_files = find_pdf_files(args.pdf_dir)
        if not pdf_files:
            logging.warning(f"No PDF files found in directory: {args.pdf_dir}")
            print(f"Warning: No PDF files found in {args.pdf_dir}. Exiting.")
            sys.exit(0)  # Exit gracefully if no PDFs found

        pdf_basenames = [os.path.basename(f) for f in pdf_files]
        print(f"Found {len(pdf_files)} PDF files: {', '.join(pdf_basenames)}")
        logging.info(f"Found {len(pdf_files)} PDF(s): {', '.join(pdf_basenames)}")

    except Exception as e:
        logging.error(f"Failed to list PDF files: {e}")
        print(f"Error: Failed to list PDF files in the directory. Check permissions.")
        sys.exit(1)

    # --- Initialize and Setup RAG System ---
    try:
        print("\nInitializing RAG system...")
        rag_system = PdfRagSystem(
            embed_model=args.embed_model,
            llm_model=args.llm_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        # Load -> Create DB -> Setup Chain
        chunks = rag_system.load_pdfs(pdf_files)  # Handles internal logging/printing
        rag_system.create_vector_db(chunks=chunks, collection_name=args.collection_name)
        rag_system.setup_rag_chain()

        print("\n=== Multi-PDF RAG System Ready ===")
        print("Type your question or '/exit' to quit.")

    except Exception as e:
        logging.critical(f"Failed to initialize RAG system: {e}", exc_info=True)
        print(f"\nError initializing RAG system: {e}")
        print("Please check your Ollama setup, model names, and file permissions.")
        sys.exit(1)

    # --- Interactive Query Loop ---
    while True:
        try:
            query = input("\nEnter your prompt: ")
            if query.lower().strip() == "/exit":
                print("Exiting PDF RAG System. Goodbye!")
                break

            result = rag_system.query(question=query)
            print("\n--- Answer ---")
            print(result)
            print("--------------")

        except Exception as e:
            logging.error(f"Error during query processing: {e}", exc_info=True)
            print(f"\nAn error occurred: {e}")
            print("Please try again or type '/exit' to quit.")
        except KeyboardInterrupt:  # Allow Ctrl+C to exit gracefully
            print("\nExiting PDF RAG System. Goodbye!")
            break


if __name__ == "__main__":
    main()
