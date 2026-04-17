import os
from typing import List
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document

from src.utils.config import RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_specific_pdfs(file_paths: List[str]) -> List[Document]:
    """
    Loads PDFs from a list of provided paths (dynamic uploading).
    High-fidelity PDF extraction using Unstructured, to partition the PDF into logical elements.
    This handles tables and multi-column layouts better than standard tools.
    """
    logger.info(f"--- Loading Pdf Elements ---")
    all_docs = []
    for path in file_paths:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            continue

        logger.info(f"Dynamically loading: {os.path.basename(path)}")
        try:
            # 'elements' mode treats each paragraph/table as a separate Document object
            # 'strategy="fast"' is good for text. Use "hi_res" for complex tables.
            loader = UnstructuredPDFLoader(
                path,
                mode="elements",
                strategy="fast",
                include_metadata=True,
                chunking_strategy="by_title",
                max_characters=1500
            )

            docs = loader.load()

            # DEBUG CHECK: Let's verify if page numbers are actually present
            if docs and "page_number" not in docs[0].metadata:
                logger.warning(f"No page_number found in {os.path.basename(path)}. "
                               "Consider switching strategy to 'hi_res' if citations fail.")

            all_docs.extend(docs)
            logger.info(f" Extraction Complete: {len(all_docs)} elements found.")

        except Exception as e:
            logger.error(f" Error during PDF extraction: {e}")

    return all_docs


if __name__ == "__main__":
    # Test implementation locally
    # Make sure to Place a sample PDF in your data/raw/ folder
    paths = [str(RAW_DATA_DIR / f) for f in ["sample_research.pdf"]]
    sample_docs = load_specific_pdfs(paths)
    if sample_docs:
        print(f"Sample Content: {sample_docs[0].page_content[:100]}...")