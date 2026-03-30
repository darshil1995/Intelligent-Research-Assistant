import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

# Import our new central config
from src.utils.config import (
    VECTOR_DB_DIR,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    RAW_DATA_DIR
)

def get_vector_store(chunks: List[Document] = None, persist_directory: str = str(VECTOR_DB_DIR)):
    """
    Creates a new Vector Store or loads an existing one.
    Refactored to use centralized project configurations.
    """

    # 1. Verify API Key using config variable
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

    # 2. Setup Embeddings using config model name
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # 3. Decision Logic: Create New vs. Load Existing
    if chunks:
        print(f"--- [VectorStore] Creating NEW store at: {persist_directory} ---")
        sanitized_chunks = filter_complex_metadata(chunks)

        vector_db = Chroma.from_documents(
            documents=sanitized_chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"Successfully indexed {len(chunks)} chunks.")
    else:
        # Check if directory exists using config logic
        if not os.path.exists(persist_directory):
            print(f"Warning: {persist_directory} does not exist. Creating empty store.")

        print(f"--- [VectorStore] Loading EXISTING store from: {persist_directory} ---")
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    return vector_db


if __name__ == "__main__":
    # Path logic handled by config
    # We just need the filename; RAW_DATA_DIR handles the rest
    TEST_FILENAME = "sample_research.pdf"
    TEST_PDF_PATH = RAW_DATA_DIR / TEST_FILENAME

    # SENIOR MOVE: Use the Path object from config to check existence
    if VECTOR_DB_DIR.exists() and any(VECTOR_DB_DIR.iterdir()):
        print("--- [Test] Vector DB found. Loading existing store. ---")
        db = get_vector_store()
    else:
        print("--- [Test] No Vector DB found. Starting ingestion pipeline. ---")
        from src.ingestion.pdf_loader import load_pdf_elements
        from src.ingestion.chunking import chunk_documents

        if TEST_PDF_PATH.exists():
            # Pass just the filename since pdf_loader is also updated to use RAW_DATA_DIR
            raw_docs = load_pdf_elements(TEST_FILENAME)
            semantic_chunks = chunk_documents(raw_docs)
            db = get_vector_store(chunks=semantic_chunks)
        else:
            print(f"Test failed: Place a PDF at {TEST_PDF_PATH}")
            exit()

    # 4. Semantic Search Test
    query = "What is Simulation?"
    print(f"\n🔍 Testing Search for: '{query}'")
    results = db.similarity_search(query, k=2)

    for i, res in enumerate(results):
        print(f"\nMatch {i + 1}: {res.page_content[:250]}...")