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
    from src.ingestion.pdf_loader import load_specific_pdfs

    # 1. Define your list of dynamic filenames
    paths = [str(RAW_DATA_DIR / f) for f in ["sample_research.pdf"]]

    # 2. Load all documents at once
    # Your new load_specific_pdfs now returns elements from ALL files in one list
    logger.info(f"Starting ingestion for {len(paths)} files...")
    all_docs = load_specific_pdfs(paths)

    if all_docs:
        # 3. Chunk everything in one go
        # The splitter processes the entire list of elements sequentially
        final_chunks = chunk_documents(all_docs)

        logger.info(f"Success! Total chunks created: {len(final_chunks)}")

        if len(final_chunks) > 0:
            print(f"--- Preview of first chunk ---\n{final_chunks[0].page_content[:200]}...")
    else:
        logger.error("No documents were loaded. Check if the file names are correct in RAW_DATA_DIR.")