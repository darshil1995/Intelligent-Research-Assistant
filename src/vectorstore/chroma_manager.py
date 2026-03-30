import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

# 1. Load environment variables from .env file immediately
# This ensures OPENAI_API_KEY is available for the embedding model
load_dotenv()

def get_vector_store(chunks: List[Document] = None, persist_directory: str = "data/vector_db"):
    """
    Creates a new Vector Store or loads an existing one.

    Why do we use this approach?:
    We use a 'Check-then-Load' pattern. This prevents accidental
    re-indexing (which costs money/time) and ensures the system
    is persistent across app restarts.
    """

    # Verify the API key exists before proceeding
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

    # 2. Setup Embeddings (The "Translator")
    # text-embedding-3-small is the current industry standard for cost/performance
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 3. Decision Logic: Create New vs. Load Existing
    if chunks:
        print(f"---Creating NEW Vector Store at: {persist_directory} ---")

        sanitized_chunks = filter_complex_metadata(chunks)

        # This takes the text chunks, sends them to OpenAI to get vectors,
        # and saves them into the Chroma SQLite database on your disk.
        vector_db = Chroma.from_documents(
            documents=sanitized_chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"Successfully indexed {len(chunks)} chunks.")
    else:
        # If no chunks are provided, we assume the DB already exists on disk
        if not os.path.exists(persist_directory):
            print(f"Warning: {persist_directory} does not exist. Creating empty store.")

        print(f"---Loading EXISTING Vector Store from: {persist_directory} ---")
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    return vector_db


if __name__ == "__main__":
    PERSIST_DIR = "data/vector_db"
    TEST_PDF = "data/raw/sample_research.pdf"

    # Check if DB exists before doing work
    if os.path.exists(PERSIST_DIR):
        print("---Vector DB found. Loading existing store to save API costs. ---")
        db = get_vector_store()  # No chunks passed = Loads existing
    else:
        print("--- No Vector DB found. Starting ingestion pipeline. ---")
        from src.ingestion.pdf_loader import load_pdf_elements
        from src.ingestion.chunking import chunk_documents

        if os.path.exists(TEST_PDF):
            raw_docs = load_pdf_elements(TEST_PDF)
            semantic_chunks = chunk_documents(raw_docs)
            db = get_vector_store(chunks=semantic_chunks)
        else:
            print(f"Test failed: Place a PDF at {TEST_PDF}")
            exit()

    # 4. Semantic Search Test
    query = "What is Simulation?"
    print(f"\n Testing Search for: '{query}'")
    results = db.similarity_search(query, k=2)

    for i, res in enumerate(results):
        print(f"\nMatch {i + 1}: {res.page_content[:250]}...")