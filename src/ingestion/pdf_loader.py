import os
from typing import List
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from src.utils.config import RAW_DATA_DIR


def load_pdf_elements(file_name: str) -> List[Document]:
    """
    High-fidelity PDF extraction using Unstructured, to partition the PDF into logical elements.
    This handles tables and multi-column layouts better than standard tools.

    Why this instead of PyPDF2?
    1. It preserves document structure (titles, list items, tables).
    2. It handles multi-column layouts using NLP-based partitioning.
    3. It captures metadata (source, page numbers) automatically.
    """

    file_path = str(RAW_DATA_DIR / file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    print(f"--- Extracting elements from: {os.path.basename(file_path)} ---")

    try:
        # 'elements' mode treats each paragraph/table as a separate Document object
        # 'strategy="fast"' is good for text. Use "hi_res" for complex tables.
        loader = UnstructuredPDFLoader(
            file_path,
            mode="elements",
            strategy="fast"
        )

        docs = loader.load()
        print(f" Extraction Complete: {len(docs)} elements found.")
        return docs

    except Exception as e:
        print(f" Error during PDF extraction: {e}")
        return []


if __name__ == "__main__":
    # Test implementation locally
    # Make sure to Place a sample PDF in your data/raw/ folder
    test_path = "sample_research.pdf"
    sample_docs = load_pdf_elements(test_path)
    if sample_docs:
        print(f"Sample Content: {sample_docs[0].page_content[:100]}...")