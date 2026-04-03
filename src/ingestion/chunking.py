from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP, RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller, overlapping chunks using a recursive strategy.

    Args:
        documents: List of Document objects from the loader.
    """
    logger.info(f"Chunking {len(documents)} document elements")

    # The recursive splitter is the 'gold standard' for general text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True  # helps with citations later,
    )

    chunks = splitter.split_documents(documents)

    logger.info(f"Chunking Complete: Created {len(chunks)} chunks.")
    return chunks


if __name__ == "__main__":
    # Test logic
    from src.ingestion.pdf_loader import load_pdf_elements
    import os

    file_name= "sample_research.pdf"
    test_path = str(RAW_DATA_DIR / file_name)
    if os.path.exists(test_path):
        docs = load_pdf_elements(test_path)
        if docs:
            final_chunks = chunk_documents(docs)
            print(f"Example Chunk 1: {final_chunks[0].page_content[:200]}")
    else:
        raise (
            logger.error(f"PDF file not found at: {test_path}"),
            FileNotFoundError(f"PDF file not found at: {test_path}")
        )