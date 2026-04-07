import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

# Centralized project configurations
from src.utils.config import (
    VECTOR_DB_DIR,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    RAW_DATA_DIR
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_vector_store(persist_directory: str = str(VECTOR_DB_DIR)):
    """
    Returns the existing Vector Store instance.
    If it doesn't exist, it returns an empty Chroma object tied to the directory.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )


def add_to_vector_store(chunks: List[Document], persist_directory: str = str(VECTOR_DB_DIR)):
    """
    Adds new chunks with unique IDs to prevent duplicates during manual uploads.
    """
    logger.info(f"--- [VectorStore] Processing {len(chunks)} chunks for the Vault ---")

    # 1. Clean metadata
    sanitized_chunks = filter_complex_metadata(chunks)

    # 2. Generate Unique IDs (Source + Index)
    # This ensures that if you upload 'report.pdf' twice, it won't duplicate entries
    ids = []
    for i, doc in enumerate(sanitized_chunks):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        ids.append(f"{source}_chunk_{i}")

    # 3. Get the store (this handles loading existing or preparing for new)
    db = get_vector_store(persist_directory)

    # 4. Add documents with IDs
    db.add_documents(sanitized_chunks, ids=ids)

    logger.info(f"Successfully synchronized {len(sanitized_chunks)} chunks.")
    return db


if __name__ == "__main__":
    from src.ingestion.pdf_loader import load_specific_pdfs
    from src.ingestion.chunking import chunk_documents

    # 1. Define files to ingest
    file_list = ["sample_research.pdf"]
    paths = [str(RAW_DATA_DIR / f) for f in file_list]

    # 2. Load
    raw_docs = load_specific_pdfs(paths)

    if not raw_docs:
        logger.error("No documents were loaded.")
    else:
        # 3. Chunk
        semantic_chunks = chunk_documents(raw_docs)

        # 4. Simplified Logic:
        # add_to_vector_store handles both "creation" and "appending"
        # because get_vector_store() returns a valid object regardless.
        db = add_to_vector_store(semantic_chunks)

        # 5. Verification Search
        query = "What are the main findings?"
        print(f"\n🔍 Testing Search for: '{query}'")
        results = db.similarity_search(query, k=3)

        for i, res in enumerate(results):
            source = res.metadata.get('source', 'Unknown')
            # Extra detail: showing the source helps verify the accumulation worked
            print(f"\n[Match {i+1}] Source: {os.path.basename(source)}")
            print(f"Content: {res.page_content[:150]}...")