from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """
    Split documents into smaller, overlapping chunks using a recursive strategy.

    Args:
        documents: List of Document objects from the loader.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks
    """
    print(f"---Chunking {len(documents)} document elements ---")

    # The recursive splitter is the 'gold standard' for general text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True  # helps with citations later,
    )

    chunks = splitter.split_documents(documents)

    print(f"Chunking Complete: Created {len(chunks)} chunks.")
    return chunks


if __name__ == "__main__":
    # Test logic
    from src.ingestion.pdf_loader import load_pdf_elements
    import os

    test_path = "data/raw/sample_research.pdf"
    if os.path.exists(test_path):
        docs = load_pdf_elements(test_path)
        if docs:
            final_chunks = chunk_documents(docs)
            print(f"Example Chunk 1: {final_chunks[0].page_content[:200]}")