import os

from langchain_core.tools import Tool

from src.utils.logger import get_logger

logger = get_logger(__name__)

from src.utils.token_counter import validate_context_budget  # Import your validator


def get_pdf_search_tool(rag_function):
    def search_with_citations(query: str):
        docs = rag_function({"question": query})
        if not docs:
            return "No relevant information found."

        formatted_context = ""
        for doc in docs:
            # Unstructured uses 'page_number'. We also check 'page' as a fallback.
            page = doc.metadata.get("page_number", doc.metadata.get("page", "N/A"))

            # Unstructured 'page_number' is usually already 1-indexed,
            # but we ensure it's readable.
            display_page = page

            source_file = os.path.basename(doc.metadata.get("source", "Research_Doc"))

            # We make the source header extremely explicit for the LLM
            new_chunk = (
                f"\n>>> SOURCE FILE: {source_file} | PAGE: {display_page} <<<\n"
                f"{doc.page_content}\n"
                f"--- END OF SEGMENT ---\n"
            )

            potential_context = formatted_context + new_chunk
            validated = validate_context_budget(potential_context, limit=2000)

            if len(validated) == len(potential_context):
                formatted_context = potential_context
            else:
                logger.warning("Context budget reached. Skipping remaining chunks.")
                break

        return formatted_context

    return Tool(
        name="PDF_Research_Search",
        func=search_with_citations,
        description="Search private docs for research with page-level citations."
    )